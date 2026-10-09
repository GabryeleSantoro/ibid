import { Link, useParams } from "@tanstack/react-router";
import { useTranslation } from "react-i18next";

import { ContextSidebar } from "@/components/shell/context-sidebar";
import { cn } from "@/lib/utils";

export const SETTINGS_SECTIONS = [
  { slug: "general" },
  { slug: "connections" },
  { slug: "appearance" },
  { slug: "chat" },
  { slug: "retrieval" },
  { slug: "performance" },
  { slug: "shortcuts" },
  { slug: "storage" },
  { slug: "updates" },
] as const;

export function SettingsSidebar() {
  const { t } = useTranslation();
  const params = useParams({ strict: false }) as { section?: string };

  return (
    <ContextSidebar title={t("nav.settings")}>
      <nav className="flex flex-col gap-0.5 pt-1">
        {SETTINGS_SECTIONS.map((section) => (
          <Link
            key={section.slug}
            to="/settings/$section"
            params={{ section: section.slug }}
            className={cn(
              "rounded-md px-2 py-1.5 text-[0.8125rem] text-sidebar-foreground transition-colors",
              "hover:bg-sidebar-accent",
              params.section === section.slug && "bg-sidebar-accent font-medium",
            )}
          >
            {t(`settings.sections.${section.slug}.label`)}
          </Link>
        ))}
      </nav>
    </ContextSidebar>
  );
}
