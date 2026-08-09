def test_dashboard_statistics_use_persisted_statuses(
    authenticated_client, user, second_user, prediction_factory, mocker
):
    prediction_factory(user, disease="Apple healthy", status="healthy")
    prediction_factory(user, disease="Bell pepper healthy", status="healthy")
    prediction_factory(user, disease="Tomato Late blight", status="diseased")
    prediction_factory(user, disease="Tomato Late blight", status="diseased")
    prediction_factory(user, disease="Potato Early blight", status="diseased")
    prediction_factory(user, disease="Low confidence healthy", status="uncertain")
    prediction_factory(user, disease="Background without leaves", status="no_leaf")
    prediction_factory(second_user, disease="Private disease", status="diseased")
    render_mock = mocker.patch(
        "app.routes.dashboard.render_template", return_value="dashboard"
    )

    response = authenticated_client.get("/")

    assert response.status_code == 200
    context = render_mock.call_args.kwargs
    assert context["total_scans"] == 7
    assert context["healthy_scans"] == 2
    assert context["diseased_scans"] == 3
    assert context["uncertain_scans"] == 1
    assert context["no_leaf_scans"] == 1
    assert (
        context["healthy_scans"]
        + context["diseased_scans"]
        + context["uncertain_scans"]
        + context["no_leaf_scans"]
        == context["total_scans"]
    )

    frequent = dict(context["frequent_diseases"])
    assert frequent == {"Tomato Late blight": 2, "Potato Early blight": 1}
    assert "Apple healthy" not in frequent
    assert "Low confidence healthy" not in frequent
    assert "Background without leaves" not in frequent
    assert "Private disease" not in frequent
