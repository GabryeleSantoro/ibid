import { useEffect, useState } from "react";
import { useTranslation } from "react-i18next";

import { Button } from "@/components/ui/button";
import { ConnectionDialog } from "@/features/settings/connection-dialog";
import {
  loadSlowState,
  recordAnswer,
  saveSlowState,
  shouldShowBanner,
  type SlowState,
} from "@/lib/slow-speed";

// Cheap, fast, non-reasoning; checked against openrouter.ai/api/v1/models.
export const OPENROUTER_SUGGESTED_MODEL = "qwen/qwen3-next-80b-a3b-instruct";

export function SlowSpeedBanner({
  done,
  tokensPerS,
}: {
  done: boolean;
  tokensPerS: number | null;
}) {
  const { t } = useTranslation();
  const [state, setState] = useState<SlowState>(loadSlowState);

  useEffect(() => {
    if (!done) return;
    setState((current) => {
      const next = recordAnswer(current, tokensPerS);
      saveSlowState(next);
      return next;
    });
    // once per finished answer
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [done]);

  if (!shouldShowBanner(state, tokensPerS, done)) return null;
  return (
    <div className="space-y-2 rounded-md border border-status-warn/30 bg-status-warn/8 px-3 py-2 text-xs text-status-warn">
      <p>{t("slowSpeed.text", { speed: Math.round(tokensPerS ?? 0) })}</p>
      <p className="opacity-80">{t("slowSpeed.privacy")}</p>
      <div className="flex gap-2">
        <ConnectionDialog
          preset={{ model_id: OPENROUTER_SUGGESTED_MODEL }}
          trigger={
            <Button size="sm" className="h-7">
              {t("slowSpeed.openrouter")}
            </Button>
          }
        />
        <Button
          variant="ghost"
          size="sm"
          className="h-7"
          onClick={() => {
            const next = { ...state, dismissed: true };
            saveSlowState(next);
            setState(next);
          }}
        >
          {t("slowSpeed.keep")}
        </Button>
      </div>
    </div>
  );
}
