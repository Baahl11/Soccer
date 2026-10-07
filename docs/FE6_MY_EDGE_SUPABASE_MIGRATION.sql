-- FE-6 My Edge subscriber persistence
-- Applied to Soccer Edge Supabase project on 2026-10-07.
-- Product state only. Never use this table as a betting-model input.

create table if not exists public.subscriber_saved_items (
  user_id uuid not null references auth.users(id) on delete cascade,
  item_key text not null check (char_length(item_key) between 1 and 220),
  item_type text not null check (item_type in ('MATCH','BET','LEAN','WATCH')),
  fixture_id bigint not null,
  market_family text null,
  market_name text null,
  selection text null,
  line numeric null,
  source_snapshot_at timestamptz null,
  payload jsonb not null default '{}'::jsonb check (jsonb_typeof(payload) = 'object'),
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  primary key (user_id, item_key)
);

comment on table public.subscriber_saved_items is
'FE6 subscriber-owned saved matches/decisions for My Edge. Product state only; never used as betting-model input.';

alter table public.subscriber_saved_items enable row level security;

drop policy if exists "subscriber_saved_items_select_own" on public.subscriber_saved_items;
create policy "subscriber_saved_items_select_own"
on public.subscriber_saved_items
for select
to authenticated
using (auth.uid() = user_id);

drop policy if exists "subscriber_saved_items_insert_own" on public.subscriber_saved_items;
create policy "subscriber_saved_items_insert_own"
on public.subscriber_saved_items
for insert
to authenticated
with check (auth.uid() = user_id);

drop policy if exists "subscriber_saved_items_update_own" on public.subscriber_saved_items;
create policy "subscriber_saved_items_update_own"
on public.subscriber_saved_items
for update
to authenticated
using (auth.uid() = user_id)
with check (auth.uid() = user_id);

drop policy if exists "subscriber_saved_items_delete_own" on public.subscriber_saved_items;
create policy "subscriber_saved_items_delete_own"
on public.subscriber_saved_items
for delete
to authenticated
using (auth.uid() = user_id);

grant select, insert, update, delete on public.subscriber_saved_items to authenticated;
