from __future__ import annotations

import json
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError

REQUEST_TIMEOUT = 20
DEFAULT_ADMIN_EMAIL = "proglppaiva@gmail.com"


class AuthConfigurationError(RuntimeError):
    pass


class AuthServiceError(RuntimeError):
    pass


def _settings() -> dict[str, str]:
    try:
        config = st.secrets["auth"]
        settings = {
            "supabase_url": str(config["supabase_url"]).rstrip("/"),
            "supabase_anon_key": str(config["supabase_anon_key"]),
            "supabase_service_role_key": str(config["supabase_service_role_key"]),
            "resend_api_key": str(config.get("resend_api_key", "")),
            "resend_from_email": str(config.get("resend_from_email", "")),
            "admin_email": str(config.get("admin_email", DEFAULT_ADMIN_EMAIL)).strip().lower(),
            "app_url": str(config.get("app_url", "")),
        }
    except (KeyError, TypeError, StreamlitSecretNotFoundError) as exc:
        raise AuthConfigurationError(
            "Configure os secrets [auth] no Streamlit Cloud e no arquivo local .streamlit/secrets.toml."
        ) from exc

    required = ("supabase_url", "supabase_anon_key", "supabase_service_role_key")
    missing = [name for name in required if not settings[name]]
    if missing:
        raise AuthConfigurationError(f"Secrets obrigatórios ausentes em [auth]: {', '.join(missing)}")
    return settings


def _request_json(url: str, payload: dict | None = None, headers: dict[str, str] | None = None) -> dict | list:
    request_headers = {"Accept": "application/json", **(headers or {})}
    body = None
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        request_headers["Content-Type"] = "application/json"
    request = Request(url, data=body, headers=request_headers, method="POST" if body is not None else "GET")
    try:
        with urlopen(request, timeout=REQUEST_TIMEOUT) as response:
            response_body = response.read()
    except HTTPError as exc:
        try:
            detail = json.loads(exc.read().decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            detail = {}
        message = detail.get("msg") or detail.get("message") or detail.get("error_description")
        if exc.code == 429:
            raise AuthServiceError("Muitas tentativas. Aguarde alguns minutos e tente novamente.") from exc
        if exc.code in (400, 401, 403):
            raise AuthServiceError(message or "Não foi possível autenticar com esses dados.") from exc
        raise AuthServiceError("O serviço de autenticação está temporariamente indisponível.") from exc
    except (URLError, TimeoutError) as exc:
        raise AuthServiceError("Não foi possível conectar ao serviço de autenticação.") from exc

    if not response_body:
        return {}
    try:
        result = json.loads(response_body.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise AuthServiceError("O serviço de autenticação retornou uma resposta inválida.") from exc
    if isinstance(result, (dict, list)):
        return result
    raise AuthServiceError("O serviço de autenticação retornou uma resposta inesperada.")


def _supabase_auth(path: str, payload: dict, settings: dict[str, str]) -> dict:
    result = _request_json(
        f"{settings['supabase_url']}/auth/v1/{path}",
        payload,
        {"apikey": settings["supabase_anon_key"]},
    )
    if not isinstance(result, dict):
        raise AuthServiceError("Resposta inválida do serviço de autenticação.")
    return result


def _admin_headers(settings: dict[str, str]) -> dict[str, str]:
    service_key = settings["supabase_service_role_key"]
    return {
        "apikey": service_key,
        "Authorization": f"Bearer {service_key}",
    }


def _profile_request(method: str, path: str, settings: dict[str, str], payload: dict | None = None) -> dict | list:
    headers = _admin_headers(settings)
    if method in {"POST", "PATCH"}:
        headers["Prefer"] = "resolution=merge-duplicates,return=representation"
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request = Request(
        f"{settings['supabase_url']}/rest/v1/{path}",
        data=body,
        headers={"Accept": "application/json", **headers, **({"Content-Type": "application/json"} if body else {})},
        method=method,
    )
    try:
        with urlopen(request, timeout=REQUEST_TIMEOUT) as response:
            response_body = response.read()
    except HTTPError as exc:
        if exc.code in (401, 403):
            raise AuthConfigurationError("A service role key não pode acessar a tabela access_requests.") from exc
        raise AuthServiceError("Não foi possível acessar a lista de pedidos de acesso.") from exc
    except (URLError, TimeoutError) as exc:
        raise AuthServiceError("Não foi possível conectar ao banco de autenticação.") from exc
    if not response_body:
        return {}
    try:
        result = json.loads(response_body.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise AuthServiceError("Resposta inválida ao consultar pedidos de acesso.") from exc
    return result if isinstance(result, (dict, list)) else {}


def _get_access_request(user_id: str, settings: dict[str, str]) -> dict | None:
    query = urlencode({"select": "user_id,email,status", "user_id": f"eq.{user_id}", "limit": "1"})
    result = _profile_request("GET", f"access_requests?{query}", settings)
    return result[0] if isinstance(result, list) and result else None


def _upsert_access_request(user_id: str, email: str, settings: dict[str, str]) -> None:
    query = urlencode({"on_conflict": "user_id"})
    _profile_request(
        "POST",
        f"access_requests?{query}",
        settings,
        {"user_id": user_id, "email": email, "status": "pending"},
    )


def _send_admin_notification(email: str, settings: dict[str, str]) -> None:
    if not settings["resend_api_key"] or not settings["resend_from_email"]:
        raise AuthConfigurationError("O aviso por e-mail requer resend_api_key e resend_from_email nos secrets [auth].")
    app_link = settings["app_url"] or "https://share.streamlit.io/"
    payload = {
        "from": settings["resend_from_email"],
        "to": [settings["admin_email"]],
        "subject": "Novo pedido de acesso ao Lotomania Quant Research Engine",
        "text": (
            f"{email} solicitou acesso ao aplicativo.\n\n"
            "Entre no app com a conta administradora para aprovar ou recusar o pedido.\n"
            f"{app_link}"
        ),
    }
    request = Request(
        "https://api.resend.com/emails",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {settings['resend_api_key']}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=REQUEST_TIMEOUT) as response:
            response.read()
    except (HTTPError, URLError, TimeoutError) as exc:
        raise AuthServiceError("O pedido foi salvo, mas não foi possível enviar o aviso por e-mail.") from exc


def _create_account(email: str, password: str, settings: dict[str, str]) -> bool:
    normalized_email = email.strip().lower()
    result = _supabase_auth("signup", {"email": normalized_email, "password": password}, settings)
    user = result.get("user") or result
    user_id = user.get("id") if isinstance(user, dict) else None
    if not user_id:
        raise AuthServiceError("O cadastro não retornou um identificador de usuário.")
    _upsert_access_request(str(user_id), normalized_email, settings)
    if not settings["resend_api_key"] or not settings["resend_from_email"]:
        return False
    try:
        _send_admin_notification(normalized_email, settings)
    except AuthServiceError:
        return False
    return True


def _login(email: str, password: str, settings: dict[str, str]) -> dict:
    return _supabase_auth(
        "token?grant_type=password",
        {"email": email.strip().lower(), "password": password},
        settings,
    )


def _review_request(user_id: str, status: str, reviewer: str, settings: dict[str, str]) -> None:
    if status not in {"approved", "denied"}:
        raise ValueError("Invalid access decision")
    query = urlencode({"user_id": f"eq.{user_id}"})
    _profile_request(
        "PATCH",
        f"access_requests?{query}",
        settings,
        {"status": status, "reviewed_by": reviewer},
    )


def _list_requests(status: str, settings: dict[str, str]) -> list[dict]:
    query = urlencode(
        {
            "select": "user_id,email,status,requested_at,reviewed_at,reviewed_by",
            "status": f"eq.{status}",
            "order": "requested_at.desc",
        }
    )
    result = _profile_request("GET", f"access_requests?{query}", settings)
    return result if isinstance(result, list) else []


def _logout() -> None:
    settings = _settings()
    access_token = st.session_state.get("auth_access_token")
    if access_token:
        request = Request(
            f"{settings['supabase_url']}/auth/v1/logout",
            headers={
                "apikey": settings["supabase_anon_key"],
                "Authorization": f"Bearer {access_token}",
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=REQUEST_TIMEOUT):
                pass
        except (HTTPError, URLError, TimeoutError):
            pass
    for key in ("auth_access_token", "auth_refresh_token", "auth_user_id", "auth_email", "auth_is_admin"):
        st.session_state.pop(key, None)
    st.rerun()


def _render_admin_requests(settings: dict[str, str]) -> None:
    st.subheader("Pedidos de acesso")
    try:
        pending = _list_requests("pending", settings)
    except (AuthConfigurationError, AuthServiceError) as exc:
        st.error(str(exc))
        return
    if not pending:
        st.info("Não há pedidos pendentes.")
    for request in pending:
        email = str(request.get("email", ""))
        user_id = str(request.get("user_id", ""))
        with st.container(border=True):
            st.write(email)
            st.caption(f"Solicitado: {request.get('requested_at', 'data indisponível')}")
            approve, deny = st.columns(2)
            if approve.button("Aprovar", key=f"approve_{user_id}", type="primary"):
                try:
                    _review_request(user_id, "approved", st.session_state.auth_email, settings)
                    st.success(f"Acesso liberado para {email}.")
                    st.rerun()
                except (AuthConfigurationError, AuthServiceError) as exc:
                    st.error(str(exc))
            if deny.button("Recusar", key=f"deny_{user_id}"):
                try:
                    _review_request(user_id, "denied", st.session_state.auth_email, settings)
                    st.success(f"Pedido de {email} recusado.")
                    st.rerun()
                except (AuthConfigurationError, AuthServiceError) as exc:
                    st.error(str(exc))


def require_authenticated() -> bool:
    try:
        settings = _settings()
    except AuthConfigurationError as exc:
        st.error("Login ainda não configurado.")
        st.info(str(exc))
        st.markdown("Configure o SQL e secrets conforme `docs/authentication.md` antes de publicar o aplicativo.")
        return False

    email = str(st.session_state.get("auth_email", "")).strip().lower()
    is_admin = bool(email and email == settings["admin_email"])
    if email and (is_admin or st.session_state.get("auth_access_token")):
        if is_admin:
            st.sidebar.success(f"Administrador: {email}")
            _render_admin_requests(settings)
            if st.sidebar.button("Sair", key="auth_logout_admin"):
                _logout()
            return True
        request = _get_access_request(str(st.session_state.get("auth_user_id", "")), settings)
        if request and request.get("status") == "approved":
            st.sidebar.caption(f"Conectado: {email}")
            if st.sidebar.button("Sair", key="auth_logout_user"):
                _logout()
            return True
        if request and request.get("status") == "denied":
            st.session_state.pop("auth_access_token", None)
            st.session_state.pop("auth_email", None)
            st.session_state.pop("auth_user_id", None)
            st.error("Este pedido de acesso foi recusado. Fale com o administrador do aplicativo.")
        else:
            st.info("Seu pedido de acesso está pendente de aprovação. Você receberá acesso quando o administrador liberar sua conta.")
            if st.sidebar.button("Sair", key="auth_logout_pending"):
                _logout()
            return False

    st.title("Acesso ao Lotomania Quant Research Engine")
    login_tab, request_tab = st.tabs(["Entrar", "Solicitar acesso"])
    with login_tab:
        with st.form("auth_login_form"):
            login_email = st.text_input("E-mail", key="auth_login_email", autocomplete="email")
            login_password = st.text_input("Senha", type="password", key="auth_login_password", autocomplete="current-password")
            submitted = st.form_submit_button("Entrar", type="primary", use_container_width=True)
        if submitted:
            if not login_email.strip() or not login_password:
                st.error("Informe e-mail e senha.")
            else:
                try:
                    session = _login(login_email, login_password, settings)
                    user = session.get("user") or {}
                    st.session_state.auth_access_token = session.get("access_token", "")
                    st.session_state.auth_refresh_token = session.get("refresh_token", "")
                    st.session_state.auth_user_id = user.get("id", "")
                    st.session_state.auth_email = str(user.get("email", login_email)).lower()
                    st.rerun()
                except AuthServiceError as exc:
                    st.error(str(exc))
    with request_tab:
        st.caption("Novas contas só acessam o app depois que o administrador aprovar o pedido.")
        with st.form("auth_request_form"):
            request_email = st.text_input("E-mail para a conta", key="auth_request_email", autocomplete="email")
            password = st.text_input("Crie uma senha", type="password", key="auth_request_password", autocomplete="new-password")
            confirm_password = st.text_input("Confirme a senha", type="password", key="auth_confirm_password", autocomplete="new-password")
            request_submitted = st.form_submit_button("Enviar pedido de acesso", use_container_width=True)
        if request_submitted:
            if not request_email.strip() or "@" not in request_email:
                st.error("Informe um endereço de e-mail válido.")
            elif len(password) < 10:
                st.error("A senha precisa ter pelo menos 10 caracteres.")
            elif password != confirm_password:
                st.error("As senhas não coincidem.")
            else:
                try:
                    email_notification_sent = _create_account(request_email, password, settings)
                    st.success("Cadastro criado e pedido enviado para aprovação.")
                    if email_notification_sent:
                        st.info("O administrador também recebeu um aviso por e-mail.")
                    else:
                        st.info("O administrador verá seu pedido na área 'Pedidos de acesso' ao entrar no app. Avisos por e-mail estão desativados.")
                    st.info("Se a confirmação de e-mail estiver ativa no Supabase, confirme também o endereço antes de entrar.")
                except (AuthConfigurationError, AuthServiceError) as exc:
                    st.error(str(exc))
    return False