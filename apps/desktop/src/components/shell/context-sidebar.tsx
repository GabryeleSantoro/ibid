import { ScrollArea } from "@/components/ui/scroll-area";
import { trafficLightGutter } from "@/lib/platform";
import { cn } from "@/lib/utils";

/**
 * The second column. Every section fills it with its own navigation
 * (chat sessions, source tree, model categories, settings groups).
 */
export function ContextSidebar({
  title,
  action,
  children,
  footer,
  className,
}: {
  title: React.ReactNode;
  action?: React.ReactNode;
  children: React.ReactNode;
  footer?: React.ReactNode;
  className?: string;
}) {
  return (
    <aside
      className={cn(
        "flex w-sidebar shrink-0 flex-col border-r border-sidebar-border bg-sidebar",
        className,
      )}
    >
      <header
        data-tauri-drag-region="deep"
        className="drag-region flex items-center justify-between gap-2 px-3 pb-2"
        style={{ paddingTop: trafficLightGutter + 12 }}
      >
        <h2 className="text-[0.8125rem] font-semibold tracking-tight text-sidebar-foreground">
          {title}
        </h2>
        {action ? (
          <div data-tauri-drag-region="false" className="no-drag">
            {action}
          </div>
        ) : null}
      </header>

      <ScrollArea className="flex-1">
        <div className="pr-4 pl-2 pb-3">{children}</div>
      </ScrollArea>

      {footer ? (
        <div className="border-t border-sidebar-border p-2">{footer}</div>
      ) : null}
    </aside>
  );
}

/** A single row in the contextual sidebar. */
export function SidebarSectionLabel({ children }: { children: React.ReactNode }) {
  return (
    <div className="px-2 pt-3 pb-1 text-[0.6875rem] font-medium tracking-wide text-muted-foreground uppercase">
      {children}
    </div>
  );
}
