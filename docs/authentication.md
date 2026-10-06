# Login and access approval

The app uses Supabase Auth for email/password accounts and a private Supabase table for approval state. Passwords are handled by Supabase Auth and are never stored by this app. Resend is optional: without it, the administrator reviews pending requests in the app's admin area.

## Configure Supabase

1. Create a Supabase project.
2. In the Supabase SQL Editor, run [`supabase/access_requests.sql`](../supabase/access_requests.sql).
3. In Authentication settings, enable email/password sign-in and email confirmation. Add the deployed Streamlit URL to the allowed redirect/site URLs.
4. Copy the project URL and the **Legacy anon** and **Legacy service_role** keys from **Project Settings → API Keys → Legacy anon, service_role API keys**. This app expects the legacy key format. The service_role key is privileged: only put it in Streamlit secrets, never in source code or a public repository.

## Optional email notifications

1. Create a Resend account and verify a sending domain.
2. Create an API key with permission to send email.
3. Use a sender address on the verified domain.
4. Add `resend_api_key` and `resend_from_email` to `[auth]` secrets. If you only have Gmail and no verified domain, omit both options; the app will still store access requests for the administrator to review.

## Local development

Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml`, then replace the placeholder values. The real `secrets.toml` is ignored by Git. Start the app with `streamlit run dashboard.py`.

## Streamlit Community Cloud

Open the app's **Settings → Secrets** page and add the TOML values from the example. Configure Supabase there; Resend values are optional. Do not commit credentials. Deploy the branch after those secrets are saved.

## How approvals work

New users register with an email and a password of at least 10 characters. The app creates a pending request. If optional Resend settings are present, it also emails the administrator; otherwise, pending requests appear in the app's admin area. The administrator signs into the app using the configured `admin_email`; pending requests and Approve/Recusar controls appear above the dashboard. All other accounts see only the pending/denied status until approved. Approval records are queried server-side with the service-role key; browser users cannot access the table directly because RLS is enabled and client roles have no table privileges.

The administrator account itself must first be created using **Solicitar acesso** (or created in Supabase Auth). Because its email is allowlisted as the administrator, it can sign in without its own approval record.