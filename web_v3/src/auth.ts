const ACCESS_KEY = "soccer_edge_access_token";
const REFRESH_KEY = "soccer_edge_refresh_token";
const AUTH_CONFIG_URL = "/app-v3-react/auth-config";
const ACCOUNT_URL = "/app/api/v2/account";

export interface AuthConfig {
  supabase_url: string | null;
  publishable_key: string | null;
  auth_configured: boolean;
  api_base: string;
}

export interface AuthState {
  authenticated: boolean;
  premiumUnlocked: boolean;
  displayRole: string;
  effectivePlan: string;
  email: string | null;
}

type Json = Record<string, unknown>;

let configPromise: Promise<AuthConfig> | null = null;

function record(value: unknown): Json {
  return value && typeof value === "object" && !Array.isArray(value) ? value as Json : {};
}

function message(data: Json, fallback: string): string {
  return String(data.error_description ?? data.msg ?? data.error ?? fallback);
}

export function storedAccessToken(): string {
  return localStorage.getItem(ACCESS_KEY) || "";
}

export function hasStoredSession(): boolean {
  return Boolean(storedAccessToken() || localStorage.getItem(REFRESH_KEY));
}

export function clearSession(): void {
  localStorage.removeItem(ACCESS_KEY);
  localStorage.removeItem(REFRESH_KEY);
}

export async function loadAuthConfig(): Promise<AuthConfig> {
  if (!configPromise) {
    configPromise = fetch(AUTH_CONFIG_URL, { cache: "no-store" })
      .then(async (response) => {
        const data = record(await response.json().catch(() => ({})));
        if (!response.ok) throw new Error(message(data, "AUTH_CONFIG_UNAVAILABLE"));
        return {
          supabase_url: typeof data.supabase_url === "string" ? data.supabase_url : null,
          publishable_key: typeof data.publishable_key === "string" ? data.publishable_key : null,
          auth_configured: data.auth_configured === true,
          api_base: typeof data.api_base === "string" ? data.api_base : "/app/api/v2",
        };
      });
  }
  return configPromise;
}

async function authFetch(path: string, body: Json): Promise<Json> {
  const config = await loadAuthConfig();
  if (!config.auth_configured || !config.supabase_url || !config.publishable_key) {
    throw new Error("AUTH_NOT_CONFIGURED");
  }
  const response = await fetch(config.supabase_url + path, {
    method: "POST",
    headers: {
      apikey: config.publishable_key,
      "Content-Type": "application/json",
    },
    body: JSON.stringify(body),
  });
  const data = record(await response.json().catch(() => ({})));
  if (!response.ok) throw new Error(message(data, "AUTH_REQUEST_FAILED"));
  return data;
}

function storeSession(data: Json): void {
  if (typeof data.access_token === "string" && data.access_token) {
    localStorage.setItem(ACCESS_KEY, data.access_token);
  }
  if (typeof data.refresh_token === "string" && data.refresh_token) {
    localStorage.setItem(REFRESH_KEY, data.refresh_token);
  }
}

export async function signIn(email: string, password: string): Promise<void> {
  const data = await authFetch("/auth/v1/token?grant_type=password", { email, password });
  if (typeof data.access_token !== "string" || !data.access_token) {
    throw new Error("AUTH_TOKEN_MISSING");
  }
  storeSession(data);
}

export async function signUp(email: string, password: string): Promise<{ signedIn: boolean; message: string }> {
  const data = await authFetch("/auth/v1/signup", { email, password });
  storeSession(data);
  const signedIn = typeof data.access_token === "string" && Boolean(data.access_token);
  return {
    signedIn,
    message: signedIn
      ? "Account created. Signing you in…"
      : "Account created. Check your email if confirmation is required.",
  };
}

export async function refreshSession(): Promise<string> {
  const refreshToken = localStorage.getItem(REFRESH_KEY) || "";
  if (!refreshToken) {
    clearSession();
    return "";
  }
  try {
    const data = await authFetch("/auth/v1/token?grant_type=refresh_token", {
      refresh_token: refreshToken,
    });
    storeSession(data);
    return typeof data.access_token === "string" ? data.access_token : "";
  } catch {
    clearSession();
    return "";
  }
}

export async function validAccessToken(): Promise<string> {
  return storedAccessToken();
}

async function accountRequest(token: string): Promise<{ response: Response; data: Json }> {
  const headers: HeadersInit = token ? { Authorization: "Bearer " + token } : {};
  const response = await fetch(ACCOUNT_URL, { headers, cache: "no-store" });
  const data = record(await response.json().catch(() => ({})));
  return { response, data };
}

export async function loadAuthState(): Promise<AuthState> {
  let token = storedAccessToken();
  let result = await accountRequest(token);

  if (result.response.status === 401 && (token || localStorage.getItem(REFRESH_KEY))) {
    token = await refreshSession();
    result = await accountRequest(token);
  }

  if (!result.response.ok) {
    if (result.response.status === 401) clearSession();
    return {
      authenticated: false,
      premiumUnlocked: false,
      displayRole: "Explorer",
      effectivePlan: "EXPLORER",
      email: null,
    };
  }

  const access = record(result.data.access);
  const user = record(result.data.user);
  return {
    authenticated: access.authenticated === true,
    premiumUnlocked: access.premium_unlocked === true,
    displayRole: String(access.display_role ?? access.effective_plan ?? (access.authenticated ? "Member" : "Explorer")),
    effectivePlan: String(access.effective_plan ?? "EXPLORER"),
    email: typeof user.email === "string" ? user.email : null,
  };
}

export function signOut(): void {
  clearSession();
}
