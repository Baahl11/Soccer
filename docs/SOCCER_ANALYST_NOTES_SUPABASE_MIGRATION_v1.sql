-- Soccer Edge analyst notebook v1. Authenticated private data, never a model input.
CREATE TABLE IF NOT EXISTS public.subscriber_match_notes (
    user_id uuid NOT NULL REFERENCES auth.users (id) ON DELETE CASCADE,
    fixture_id bigint NOT NULL CHECK (fixture_id > 0),
    note_text text NOT NULL CHECK (char_length(note_text) BETWEEN 1 AND 2000),
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (user_id, fixture_id)
);
COMMENT ON TABLE public.subscriber_match_notes IS
    'Private analyst hypotheses, product-only. Never used as a sport model, calibrated prediction, or automated value signal.';
ALTER TABLE public.subscriber_match_notes ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "subscriber_match_notes_read_own" ON public.subscriber_match_notes;
CREATE POLICY "subscriber_match_notes_read_own"
    ON public.subscriber_match_notes FOR SELECT TO authenticated
    USING ((SELECT auth.uid()) = user_id);

DROP POLICY IF EXISTS "subscriber_match_notes_insert_own" ON public.subscriber_match_notes;
CREATE POLICY "subscriber_match_notes_insert_own"
    ON public.subscriber_match_notes FOR INSERT TO authenticated
    WITH CHECK ((SELECT auth.uid()) = user_id);

DROP POLICY IF EXISTS "subscriber_match_notes_update_own" ON public.subscriber_match_notes;
CREATE POLICY "subscriber_match_notes_update_own"
    ON public.subscriber_match_notes FOR UPDATE TO authenticated
    USING ((SELECT auth.uid()) = user_id)
    WITH CHECK ((SELECT auth.uid()) = user_id);

DROP POLICY IF EXISTS "subscriber_match_notes_delete_own" ON public.subscriber_match_notes;
CREATE POLICY "subscriber_match_notes_delete_own"
    ON public.subscriber_match_notes FOR DELETE TO authenticated
    USING ((SELECT auth.uid()) = user_id);

GRANT SELECT, INSERT, UPDATE, DELETE ON public.subscriber_match_notes TO authenticated;
REVOKE ALL ON public.subscriber_match_notes FROM anon;
