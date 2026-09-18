"""Compatibility launcher; no Streamlit side effects during test collection."""
if __name__ == "__main__":
    from scripts.doctor import main
    raise SystemExit(main())
