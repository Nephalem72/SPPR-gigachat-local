from sppr_colab.ui import _sources_view, create_demo


def test_sources_are_rendered_for_user_and_cases_can_be_opened() -> None:
    rows, choices = _sources_view(
        [
            {"id": "L1", "source": "УК РФ", "score": 0.75},
            {
                "id": "C1",
                "case_number": "1-1/2024",
                "court": "Районный суд",
                "date": "01.01.2024",
                "score": 0.61,
            },
        ]
    )

    assert rows[0] == ["L1", "Правовой материал", "УК РФ", "", "75.0%"]
    assert rows[1][1] == "Судебное дело"
    assert choices == [("C1 · Дело 1-1/2024", "1-1/2024")]


def test_gradio_demo_builds() -> None:
    demo = create_demo()
    assert demo.title == "СППР"
