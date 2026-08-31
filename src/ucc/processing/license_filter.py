"""Stage 6 — license filtering.

Normalizes detected license strings to SPDX-ish ids, classifies each record
into permissive / weak-copyleft / strong-copyleft / unknown (the most
restrictive detected license wins) and applies the configured action
(keep / flag / drop). Provenance is always retained; dropped records land in
the excluded-report with their license verdict.

The pipeline never assumes source data can automatically be redistributed —
see the README's redistribution notes and the generated dataset card.
"""

from __future__ import annotations

from ucc import constants as C
from ucc.processing.base import ShardContext, Stage

# lowercase alias -> canonical SPDX id
LICENSE_ALIASES = {
    "mit": "MIT", "mit license": "MIT", "expat": "MIT",
    "apache-2.0": "Apache-2.0", "apache 2.0": "Apache-2.0", "apache2": "Apache-2.0",
    "apache license 2.0": "Apache-2.0",
    "bsd-2-clause": "BSD-2-Clause", "bsd-3-clause": "BSD-3-Clause",
    "bsd-3-clause-clear": "BSD-3-Clause-Clear", "0bsd": "0BSD",
    "isc": "ISC", "unlicense": "Unlicense", "the unlicense": "Unlicense",
    "cc0-1.0": "CC0-1.0", "cc-by-4.0": "CC-BY-4.0", "cc-by-3.0": "CC-BY-3.0",
    "zlib": "Zlib", "wtfpl": "WTFPL", "artistic-2.0": "Artistic-2.0",
    "python-2.0": "Python-2.0", "psf-2.0": "PSF-2.0", "postgresql": "PostgreSQL",
    "ncsa": "NCSA", "ms-pl": "MS-PL", "bsl-1.0": "BSL-1.0", "unicode-dfs-2016": "Unicode-DFS-2016",
    "mpl-2.0": "MPL-2.0",
    "lgpl-2.1": "LGPL-2.1-only", "lgpl-2.1-only": "LGPL-2.1-only",
    "lgpl-2.1-or-later": "LGPL-2.1-or-later", "lgpl-3.0": "LGPL-3.0-only",
    "lgpl-3.0-only": "LGPL-3.0-only", "lgpl-3.0-or-later": "LGPL-3.0-or-later",
    "epl-1.0": "EPL-1.0", "epl-2.0": "EPL-2.0", "cddl-1.0": "CDDL-1.0",
    "cc-by-sa-4.0": "CC-BY-SA-4.0", "cc-by-sa-3.0": "CC-BY-SA-3.0",
    "osl-3.0": "OSL-3.0", "eupl-1.2": "EUPL-1.2",
    "gpl-2.0": "GPL-2.0-only", "gpl-2.0-only": "GPL-2.0-only",
    "gpl-2.0-or-later": "GPL-2.0-or-later", "gpl-3.0": "GPL-3.0-only",
    "gpl-3.0-only": "GPL-3.0-only", "gpl-3.0-or-later": "GPL-3.0-or-later",
    "agpl-3.0": "AGPL-3.0-only", "agpl-3.0-only": "AGPL-3.0-only",
    "agpl-3.0-or-later": "AGPL-3.0-or-later", "sspl-1.0": "SSPL-1.0",
}

PERMISSIVE = {
    "MIT", "Apache-2.0", "BSD-2-Clause", "BSD-3-Clause", "BSD-3-Clause-Clear",
    "0BSD", "ISC", "Unlicense", "CC0-1.0", "CC-BY-4.0", "CC-BY-3.0", "Zlib",
    "WTFPL", "Artistic-2.0", "Python-2.0", "PSF-2.0", "PostgreSQL", "NCSA",
    "MS-PL", "BSL-1.0", "Unicode-DFS-2016",
}
WEAK_COPYLEFT = {
    "MPL-2.0", "LGPL-2.1-only", "LGPL-2.1-or-later", "LGPL-3.0-only",
    "LGPL-3.0-or-later", "EPL-1.0", "EPL-2.0", "CDDL-1.0", "CC-BY-SA-4.0",
    "CC-BY-SA-3.0", "OSL-3.0", "EUPL-1.2",
}
STRONG_COPYLEFT = {
    "GPL-2.0-only", "GPL-2.0-or-later", "GPL-3.0-only", "GPL-3.0-or-later",
    "AGPL-3.0-only", "AGPL-3.0-or-later", "SSPL-1.0",
}

_RESTRICTIVENESS = {
    C.LIC_PERMISSIVE: 0,
    C.LIC_WEAK_COPYLEFT: 1,
    C.LIC_UNKNOWN: 2,
    C.LIC_STRONG_COPYLEFT: 3,
}


def normalize_license(raw: str) -> str:
    key = raw.strip().lower()
    return LICENSE_ALIASES.get(key, raw.strip())


def classify_license(spdx: str) -> str:
    if spdx in PERMISSIVE:
        return C.LIC_PERMISSIVE
    if spdx in WEAK_COPYLEFT:
        return C.LIC_WEAK_COPYLEFT
    if spdx in STRONG_COPYLEFT:
        return C.LIC_STRONG_COPYLEFT
    return C.LIC_UNKNOWN


class LicenseFilterStage(Stage):
    name = "license_filter"

    def run(self, rows: list[dict], ctx: ShardContext) -> list[dict]:
        lcfg = ctx.cfg.licenses
        actions = {
            C.LIC_PERMISSIVE: lcfg.get("permissive_action", "keep"),
            C.LIC_WEAK_COPYLEFT: lcfg.get("weak_copyleft_action", "flag"),
            C.LIC_STRONG_COPYLEFT: lcfg.get("strong_copyleft_action", "drop"),
            C.LIC_UNKNOWN: lcfg.get("unknown_action", "flag"),
        }
        source_level_permissive = set(lcfg.get("source_level_permissive") or [])
        source = ctx.shard["source_dataset"]

        out: list[dict] = []
        for rec in rows:
            normalized = sorted({normalize_license(x) for x in rec.get("detected_licenses") or []})
            if normalized:
                status = max(
                    (classify_license(x) for x in normalized),
                    key=lambda s: _RESTRICTIVENESS[s],
                )
                rec["license"] = ", ".join(normalized)
                rec["detected_licenses"] = normalized
            elif source in source_level_permissive:
                # Upstream dataset is already license-filtered to permissive.
                status = C.LIC_PERMISSIVE
                rec["license"] = None
                rec["quality_flags"] = sorted(
                    set(rec.get("quality_flags") or []) | {"license_source_level"}
                )
            else:
                status = C.LIC_UNKNOWN
                rec["license"] = None

            rec["license_status"] = status
            ctx.bump(f"license.{status}")

            action = actions.get(status, "flag")
            if action == "drop":
                ctx.exclude(rec, f"license_{status}", detail=rec.get("license") or "")
                continue
            if action == "flag" and status != C.LIC_PERMISSIVE:
                rec["quality_flags"] = sorted(
                    set(rec.get("quality_flags") or []) | {f"license_{status}"}
                )
            out.append(rec)

        ctx.bump("license.records_out", len(out))
        return out
