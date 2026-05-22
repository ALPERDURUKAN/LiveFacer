import tkinter


_PATCH_APPLIED = False


def apply_patch() -> None:
    global _PATCH_APPLIED
    if _PATCH_APPLIED:
        return

    original_init = tkinter.Tk.__init__

    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        try:
            self.tk.eval(
                """
                if {[info commands ::tk::ScreenChanged] == ""} {
                    proc ::tk::ScreenChanged {args} {
                        return
                    }
                }
                """
            )
        except Exception:
            # Keep startup resilient even if Tcl patching fails.
            pass

    tkinter.Tk.__init__ = patched_init
    _PATCH_APPLIED = True


apply_patch()
