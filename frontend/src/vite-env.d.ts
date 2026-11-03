/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** "browser" builds the edition that runs the whole app in the page (Vercel); unset builds the server edition. */
  readonly VITE_ENGINE?: "browser";
}
