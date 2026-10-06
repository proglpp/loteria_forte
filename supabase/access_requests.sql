create table if not exists public.access_requests (
    user_id uuid primary key references auth.users (id) on delete cascade,
    email text not null,
    status text not null default 'pending'
        check (status in ('pending', 'approved', 'denied')),
    requested_at timestamptz not null default now(),
    reviewed_at timestamptz,
    reviewed_by text
);

create index if not exists access_requests_status_requested_at_idx
    on public.access_requests (status, requested_at desc);

alter table public.access_requests enable row level security;
revoke all on table public.access_requests from anon, authenticated;
grant select, insert, update, delete on table public.access_requests to service_role;

create or replace function public.set_access_request_reviewed_at()
returns trigger
language plpgsql
as $$
begin
    if new.status is distinct from old.status and new.status <> 'pending' then
        new.reviewed_at = now();
    end if;
    return new;
end;
$$;

drop trigger if exists access_requests_reviewed_at on public.access_requests;
create trigger access_requests_reviewed_at
    before update on public.access_requests
    for each row
    execute function public.set_access_request_reviewed_at();