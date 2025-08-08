def test_import_and_version():
    import slick

    assert isinstance(slick.__version__, str)
    assert len(slick.__version__) > 0
