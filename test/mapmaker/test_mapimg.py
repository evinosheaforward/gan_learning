from gan.mapmaker import mapimg


def test_load_data():
    """Load whole dataset, long test"""
    data = mapimg.load_images("data/dnd_maps")
    data = data.cpu()
    print(data)
    print(data.size())
    assert data.size() == (1, 3, 128, 128)
