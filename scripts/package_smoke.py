from __future__ import annotations

from importlib.resources import files

import icframe


def main() -> None:
    pack_root = files("icframe.domain_packs")
    static_root = files("icframe.ui") / "static"
    assert (pack_root / "public_goods" / "pack.toml").is_file()
    assert (pack_root / "public_goods" / "spec.toml").is_file()
    assert (static_root / "index.html").is_file()
    assert (static_root / "app.js").is_file()
    assert icframe.load_domain_pack("public_goods").id == "public_goods"


if __name__ == "__main__":
    main()
