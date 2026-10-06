import auth_ui


def test_create_account_saves_pending_request_without_resend(monkeypatch):
    calls = []
    settings = {
        "supabase_url": "https://example.supabase.co",
        "supabase_anon_key": "anon",
        "supabase_service_role_key": "service-role",
        "resend_api_key": "",
        "resend_from_email": "",
        "admin_email": "admin@example.com",
        "app_url": "https://example.streamlit.app",
    }
    monkeypatch.setattr(
        auth_ui,
        "_supabase_auth",
        lambda path, payload, current_settings: {"user": {"id": "user-1"}},
    )
    monkeypatch.setattr(
        auth_ui,
        "_upsert_access_request",
        lambda user_id, email, current_settings: calls.append((user_id, email)),
    )
    monkeypatch.setattr(
        auth_ui,
        "_send_admin_notification",
        lambda email, current_settings: (_ for _ in ()).throw(AssertionError("Resend must not be called")),
    )

    notification_sent = auth_ui._create_account(" New.User@Example.com ", "a-strong-password", settings)

    assert calls == [("user-1", "new.user@example.com")]
    assert notification_sent is False


def test_create_account_keeps_request_if_optional_email_fails(monkeypatch):
    calls = []
    settings = {
        "supabase_url": "https://example.supabase.co",
        "supabase_anon_key": "anon",
        "supabase_service_role_key": "service-role",
        "resend_api_key": "re_test",
        "resend_from_email": "App <access@example.com>",
        "admin_email": "admin@example.com",
        "app_url": "https://example.streamlit.app",
    }
    monkeypatch.setattr(
        auth_ui,
        "_supabase_auth",
        lambda path, payload, current_settings: {"user": {"id": "user-2"}},
    )
    monkeypatch.setattr(
        auth_ui,
        "_upsert_access_request",
        lambda user_id, email, current_settings: calls.append((user_id, email)),
    )

    def fail_notification(email, current_settings):
        raise auth_ui.AuthServiceError("mail service unavailable")

    monkeypatch.setattr(auth_ui, "_send_admin_notification", fail_notification)

    notification_sent = auth_ui._create_account("user@example.com", "a-strong-password", settings)

    assert calls == [("user-2", "user@example.com")]
    assert notification_sent is False