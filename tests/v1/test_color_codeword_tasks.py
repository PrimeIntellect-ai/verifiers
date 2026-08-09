from color_codeword import taskset


def test_color_images_are_rendered_once(monkeypatch) -> None:
    calls: list[tuple[int, int, int]] = []
    render = taskset.image_data_url

    def counted(image) -> str:
        calls.append(image.getpixel((0, 0)))
        return render(image)

    monkeypatch.setattr(taskset, "image_data_url", counted)
    tasks = list(taskset.ColorCodewordTaskset(taskset.ColorCodewordConfig()).head(100))

    assert len(tasks) == 100
    assert calls == list(taskset.COLOR_RGB.values())
