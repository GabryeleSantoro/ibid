import { useEffect, useRef, useState } from "react";
import { ArrowUpIcon, SquareIcon } from "lucide-react";
import { useTranslation } from "react-i18next";

import { Button } from "@/components/ui/button";
import { IconTooltip } from "@/components/ui/tooltip";
import { COMPOSER_FOCUS_EVENT } from "@/lib/shortcuts";
import { cn } from "@/lib/utils";

const MAX_ROWS_PX = 180;

export function Composer({
  onSend,
  onStop,
  streaming,
  disabled,
  placeholder,
  children,
}: {
  onSend: (text: string) => void;
  onStop: () => void;
  streaming: boolean;
  disabled?: boolean;
  placeholder?: string;
  /** Filter chips and the mode selector sit above the input. */
  children?: React.ReactNode;
}) {
  const { t } = useTranslation();
  const [value, setValue] = useState("");
  const ref = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    const element = ref.current;
    if (!element) return;
    element.style.height = "auto";
    element.style.height = `${Math.min(element.scrollHeight, MAX_ROWS_PX)}px`;
  }, [value]);

  useEffect(() => {
    // Deferred a tick: after a route change the textarea may not be mounted yet.
    const focus = () => setTimeout(() => ref.current?.focus(), 0);
    window.addEventListener(COMPOSER_FOCUS_EVENT, focus);
    return () => window.removeEventListener(COMPOSER_FOCUS_EVENT, focus);
  }, []);

  const submit = () => {
    const text = value.trim();
    if (!text || streaming || disabled) return;
    onSend(text);
    setValue("");
  };

  return (
    <div className="bg-linear-to-t from-background via-background/90 to-transparent px-5 pb-4 pt-6">
      <div className="mx-auto w-full max-w-3xl">
        {children ? <div className="mb-2 flex flex-wrap gap-1.5">{children}</div> : null}

        <div
          className={cn(
            "glass flex items-end gap-2 rounded-2xl border border-input p-2 shadow-lg shadow-primary/5 transition-colors",
            "focus-within:border-ring focus-within:ring-[3px] focus-within:ring-ring/25",
            disabled && "opacity-60",
          )}
        >
          <textarea
            ref={ref}
            rows={1}
            value={value}
            disabled={disabled}
            placeholder={placeholder ?? t("chat.composerPlaceholder")}
            onChange={(event) => setValue(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter" && !event.shiftKey) {
                event.preventDefault();
                submit();
              }
            }}
            className="max-h-45 flex-1 resize-none bg-transparent px-2 py-1.5 text-sm outline-none placeholder:text-muted-foreground"
          />
          {streaming ? (
            <IconTooltip label={t("chat.stop")}>
              <Button
                size="icon"
                variant="secondary"
                className="size-8"
                aria-label={t("chat.stop")}
                onClick={onStop}
              >
                <SquareIcon className="size-3.5 fill-current" />
              </Button>
            </IconTooltip>
          ) : (
            <IconTooltip label={t("chat.send")}>
              <Button
                size="icon"
                className="size-8"
                aria-label={t("chat.send")}
                disabled={!value.trim() || disabled}
                onClick={submit}
              >
                <ArrowUpIcon className="size-4" />
              </Button>
            </IconTooltip>
          )}
        </div>

        <p className="mt-1.5 px-1 text-[0.6875rem] text-muted-foreground">
          {t("chat.composerHint")}
        </p>
      </div>
    </div>
  );
}
