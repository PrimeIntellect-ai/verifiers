from color_codeword_v1 import taskset


def test_color_images_are_rendered_once(monkeypatch) -> None:
    calls: list[str] = []
    render = taskset.color_data_url

    def counted(color: str, size: int = 100) -> str:
        calls.append(color)
        return render(color, size)

    monkeypatch.setattr(taskset, "color_data_url", counted)
    tasks = taskset.ColorCodewordTaskset(taskset.ColorCodewordConfig()).select(
        num_tasks=100
    )

    assert len(tasks) == 100
    assert calls == list(taskset.COLOR_MAP)
