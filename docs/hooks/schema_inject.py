# ------------------------------------------------------------------------
# re-ID
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""MkDocs hook: per-page TechArticle and BreadcrumbList JSON-LD.

Adapted from roboflow/trackers docs/hooks/schema_inject.py, keeping the parts that need no page content: a
TechArticle for every page with a ``description`` in its front matter, and breadcrumbs from the navigation. The
serialized JSON goes into ``page.meta`` and docs/overrides/main.html emits it.
"""

import json

# Canonical Roboflow organization @id, shared across Roboflow docs; must match docs/overrides/main.html.
ORG_ID = "https://roboflow.com/#organization"


def _build_breadcrumbs(page, config, nav):  # type: ignore[no-untyped-def]
    """Build BreadcrumbList JSON-LD from navigation hierarchy.

    Returns None if the page is at the root level (no meaningful breadcrumb) or if the page is the homepage (to avoid
    "Home > Home > ..." duplication).
    """
    # Skip breadcrumbs for the homepage to avoid "Home > Home > ..." duplication.
    if page.file.src_path == "index.md":
        return None

    site_url = config.get("site_url", "").rstrip("/")

    # Walk the nav tree to find the path of sections leading to this page.
    crumbs = [{"name": "Home", "url": site_url + "/"}]

    def _resolve_nav_item_url(item):  # type: ignore[no-untyped-def]
        """Return an absolute URL for a nav item when it maps to a concrete page."""
        raw_url = getattr(item, "url", None)
        if not raw_url and hasattr(item, "file") and item.file:
            raw_url = getattr(item.file, "url", None)

        if not raw_url:
            return ""
        if raw_url.startswith(("http://", "https://")):
            return raw_url
        if raw_url.startswith("/"):
            return site_url + raw_url
        return site_url + "/" + raw_url.lstrip("/")

    def _find_in_nav(items, path):  # type: ignore[no-untyped-def]
        """Recursively search nav for the page, building the path of sections."""
        for item in items:
            if hasattr(item, "children") and item.children:
                section_url = _resolve_nav_item_url(item)
                appended = False
                if section_url:
                    path.append({"name": item.title, "url": section_url})
                    appended = True
                if _find_in_nav(item.children, path):
                    return True
                if appended:
                    path.pop()
            elif hasattr(item, "file") and item.file and item.file.src_path == page.file.src_path:
                return True
        return False

    section_path: list[dict[str, str]] = []
    _find_in_nav(nav.items, section_path)

    if not section_path:
        return None

    crumbs.extend(section_path)
    crumbs.append({"name": page.title or "", "url": page.canonical_url or ""})

    items = []
    for i, crumb in enumerate(crumbs, start=1):
        items.append(
            {
                "@type": "ListItem",
                "position": i,
                "name": crumb["name"],
                **({"item": crumb["url"]} if crumb["url"] else {}),
            }
        )

    return {
        "@context": "https://schema.org",
        "@type": "BreadcrumbList",
        "itemListElement": items,
    }


def on_page_context(context, page, config, nav):  # type: ignore[no-untyped-def]
    """Store TechArticle and BreadcrumbList JSON-LD for the page in page.meta."""
    description = (page.meta or {}).get("description", "")
    title = page.title or ""
    canonical = page.canonical_url or ""

    if page.meta is None:
        page.meta = {}  # type: ignore[assignment]

    site_url = config.get("site_url", "").rstrip("/")
    logo = {"@type": "ImageObject", "url": f"{site_url}/assets/roboflow-logo.svg"}

    if description:
        article = {
            "@context": "https://schema.org",
            "@type": "TechArticle",
            "headline": title,
            "description": description,
            "url": canonical,
            "mainEntityOfPage": {"@type": "WebPage", "@id": canonical},
            "image": logo,
            "author": {"@type": "Organization", "@id": ORG_ID, "name": "Roboflow"},
            "publisher": {"@type": "Organization", "@id": ORG_ID, "name": "Roboflow", "logo": logo},
            "speakable": {"@type": "SpeakableSpecification", "cssSelector": ["h1", ".md-content p:first-of-type"]},
        }
        page.meta["json_ld_article"] = json.dumps(article, ensure_ascii=False, indent=2)

    breadcrumbs = _build_breadcrumbs(page, config, nav)
    if breadcrumbs:
        page.meta["json_ld_breadcrumbs"] = json.dumps(breadcrumbs, ensure_ascii=False, indent=2)

    return context
