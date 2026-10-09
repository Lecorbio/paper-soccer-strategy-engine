"""Use the current bound controller at launch and before new research claims."""
from __future__ import annotations

import importlib.util

from tools import rank_two_focused_campaign_v1 as campaign


def checked(plan, action, launch=False):
    current = campaign.read(plan["campaign_current"])
    if (current.get("focused_activation") != plan["activation"]
            or current.get("owner_thread_id") != plan["owner_thread_id"]
            or current.get("ownership_status") != "active"):
        raise PermissionError("bound work ownership or activation changed")
    specification = importlib.util.spec_from_file_location("focused_current_work_guard",
                                                          campaign.verify(current["control_guard"]))
    guard = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(guard)
    reference = current.get("focused_usage_observation")
    if reference is None or not hasattr(guard, "actual_usage"):
        raise PermissionError("actual usage receipt and current focused controller required")
    # Every launch requires a fresh real response. During already admitted work,
    # each wake refreshes the receipt; pause/deadline/incident changes apply at
    # the next boundary before another numerical root or teacher parent claim.
    used = guard.actual_usage(campaign.read(campaign.verify(reference)), fresh=launch)
    guard.validate(current, campaign.read(campaign.verify(current["closures"])),
                   plan["owner_thread_id"], action, used)
    return current
