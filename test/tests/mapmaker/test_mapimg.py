from gan.mapmaker import mapimg


def test_load_data():
    data = mapimg.load_data("data/dnd_maps", batch_size=1)

    assert data.size() == (1, 3, 800, 800)
