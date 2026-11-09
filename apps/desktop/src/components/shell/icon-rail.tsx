import { Link, useRouterState } from "@tanstack/react-router";
import { useTranslation } from "react-i18next";
import {
  ActivityIcon,
  BookTextIcon,
  BoxesIcon,
  FlaskConicalIcon,
  LibraryIcon,
  ScrollTextIcon,
  FilePenLineIcon,
  MessagesSquareIcon,
  NetworkIcon,
  SettingsIcon,
  type LucideIcon,
} from "lucide-react";

import { ThemeToggle } from "@/components/shell/theme-provider";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { trafficLightGutter } from "@/lib/platform";
import { cn } from "@/lib/utils";

type RailItem = {
  to: string;
  labelKey: string;
  icon: LucideIcon;
  /** Matches nested routes, e.g. /reader/* belongs to Library. */
  match: string[];
  devOnly?: boolean;
};

const PRIMARY: RailItem[] = [
  { to: "/chat", labelKey: "nav.chat", icon: MessagesSquareIcon, match: ["/chat", "/"] },
  { to: "/library", labelKey: "nav.library", icon: LibraryIcon, match: ["/library", "/reader"] },
  { to: "/topics", labelKey: "nav.topics", icon: NetworkIcon, match: ["/topics"] },
  { to: "/convert", labelKey: "nav.convert", icon: FilePenLineIcon, match: ["/convert"] },
  { to: "/book", labelKey: "nav.book", icon: BookTextIcon, match: ["/book"] },
  { to: "/models", labelKey: "nav.models", icon: BoxesIcon, match: ["/models"] },
  { to: "/settings", labelKey: "nav.settings", icon: SettingsIcon, match: ["/settings"] },
];

const SECONDARY: RailItem[] = [
  { to: "/logs", labelKey: "nav.logs", icon: ScrollTextIcon, match: ["/logs"] },
  { to: "/diagnostics", labelKey: "nav.diagnostics", icon: ActivityIcon, match: ["/diagnostics"] },
  { to: "/eval", labelKey: "nav.eval", icon: FlaskConicalIcon, match: ["/eval"], devOnly: true },
];

function RailButton({ item, active }: { item: RailItem; active: boolean }) {
  const { t } = useTranslation();
  const label = t(item.labelKey);
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <Link
          to={item.to}
          aria-label={label}
          aria-current={active ? "page" : undefined}
          data-tour={item.to}
          data-tauri-drag-region="false"
          className={cn(
            "no-drag relative grid size-9 place-items-center rounded-lg transition-[background-color,color,transform] duration-200",
            "text-rail-foreground hover:bg-sidebar-accent hover:text-foreground",
            active && "bg-primary/12 text-primary",
          )}
        >
          {/* Active marker rides the rail edge so the icon itself stays unstyled. */}
          <span
            className={cn(
              "absolute -left-2 h-5 w-0.75 rounded-full bg-primary transition-opacity",
              active ? "opacity-100" : "opacity-0",
            )}
          />
          <item.icon className="size-4.5" strokeWidth={1.75} />
        </Link>
      </TooltipTrigger>
      <TooltipContent side="right">{label}</TooltipContent>
    </Tooltip>
  );
}

export function IconRail({ devMode = false }: { devMode?: boolean }) {
  const pathname = useRouterState({ select: (s) => s.location.pathname });
  const isActive = (item: RailItem) =>
    item.match.some((m) => (m === "/" ? pathname === "/" : pathname.startsWith(m)));

  return (
    <nav
      data-tauri-drag-region="deep"
      className="drag-region flex w-rail shrink-0 flex-col items-center gap-1 border-r border-sidebar-border bg-rail pb-3"
      style={{ paddingTop: trafficLightGutter + 12 }}
    >
      {PRIMARY.map((item) => (
        <RailButton key={item.to} item={item} active={isActive(item)} />
      ))}

      <div className="flex-1" />

      {SECONDARY.filter((item) => !item.devOnly || devMode).map((item) => (
        <RailButton key={item.to} item={item} active={isActive(item)} />
      ))}
      <div data-tauri-drag-region="false" className="no-drag">
        <ThemeToggle />
      </div>
    </nav>
  );
}
