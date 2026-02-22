#!/usr/bin/env python3
"""Floating status bar overlay for Diet Assistant (native macOS).

Runs as a subprocess. Reads single-line commands from stdin:
    recording       — show "Listening..." (red accent)
    processing      — show "Processing..." (yellow accent)
    typing          — show text input field (blue accent)
    result:<text>   — show assistant response briefly, then hide
    hide            — hide the bar
    quit            — exit

When in typing mode, user input is sent back to the parent process
via stdout as:  typed:<text>
"""

import sys
import threading

import AppKit
import objc
from Foundation import NSObject, NSTimer


# ── Colors ────────────────────────────────────────────────────────────────

STATES = {
    "recording":  {"accent": (1.0, 0.23, 0.19, 1.0), "text": "Listening..."},
    "processing": {"accent": (1.0, 0.8, 0.0, 1.0),   "text": "Processing..."},
    "result":     {"accent": (0.2, 0.78, 0.35, 1.0),  "text": ""},
    "typing":     {"accent": (0.25, 0.52, 1.0, 1.0),  "text": ""},
}

BAR_WIDTH = 340
BAR_HEIGHT = 36
FADE_AFTER_SEC = 3.0


class OverlayView(AppKit.NSView):
    """Custom view that draws the dark rounded bar with accent dot + text."""

    def initWithFrame_(self, frame):
        self = objc.super(OverlayView, self).initWithFrame_(frame)
        if self is None:
            return None
        self._accent_color = AppKit.NSColor.redColor()
        self._text = "Listening..."
        return self

    def setAccentColor_(self, color):
        self._accent_color = color
        self.setNeedsDisplay_(True)

    def setText_(self, text):
        self._text = text
        self.setNeedsDisplay_(True)

    def drawRect_(self, rect):
        bg = AppKit.NSColor.colorWithCalibratedRed_green_blue_alpha_(0.1, 0.1, 0.1, 0.92)
        path = AppKit.NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(
            self.bounds(), 10, 10
        )
        bg.setFill()
        path.fill()

        # Accent dot
        dot_rect = AppKit.NSMakeRect(14, (BAR_HEIGHT - 10) / 2, 10, 10)
        dot_path = AppKit.NSBezierPath.bezierPathWithOvalInRect_(dot_rect)
        self._accent_color.setFill()
        dot_path.fill()

        # Text (only drawn when not in typing mode — typing mode uses NSTextField)
        if self._text:
            attrs = {
                AppKit.NSFontAttributeName: AppKit.NSFont.systemFontOfSize_weight_(14, 0.3),
                AppKit.NSForegroundColorAttributeName: AppKit.NSColor.whiteColor(),
            }
            text_str = AppKit.NSAttributedString.alloc().initWithString_attributes_(
                self._text, attrs
            )
            text_rect = AppKit.NSMakeRect(32, (BAR_HEIGHT - 18) / 2, BAR_WIDTH - 44, 20)
            text_str.drawInRect_(text_rect)


class KeyableWindow(AppKit.NSWindow):
    """Borderless window that can become key window (needed for text input)."""

    def canBecomeKeyWindow(self):
        return True


class TextFieldDelegate(NSObject):
    """Handles Enter key press in the text field."""

    controller = objc.ivar()

    def controlTextDidEndEditing_(self, notification):
        """Called when the user presses Enter."""
        text_field = notification.object()
        text = text_field.stringValue()
        if text and self.controller:
            # Send typed text to parent process via stdout
            sys.stdout.write(f"typed:{text}\n")
            sys.stdout.flush()
            # Switch to processing state
            self.controller._enter_processing_after_type()


class OverlayController(NSObject):
    """Main controller for the overlay window."""

    def init(self):
        self = objc.super(OverlayController, self).init()
        if self is None:
            return None
        self._visible = False
        self._hide_timer = None
        self._typing_mode = False
        self._text_field = None
        self._text_delegate = None
        return self

    def setup(self):
        app = AppKit.NSApplication.sharedApplication()
        app.setActivationPolicy_(AppKit.NSApplicationActivationPolicyAccessory)

        screen = AppKit.NSScreen.mainScreen().frame()
        x = (screen.size.width - BAR_WIDTH) / 2
        y = 80

        frame = AppKit.NSMakeRect(x, y, BAR_WIDTH, BAR_HEIGHT)
        self.window = KeyableWindow.alloc().initWithContentRect_styleMask_backing_defer_(
            frame,
            AppKit.NSWindowStyleMaskBorderless,
            AppKit.NSBackingStoreBuffered,
            False,
        )
        self.window.setLevel_(AppKit.NSFloatingWindowLevel)
        self.window.setOpaque_(False)
        self.window.setBackgroundColor_(AppKit.NSColor.clearColor())
        self.window.setHasShadow_(True)
        self.window.setIgnoresMouseEvents_(True)

        # Global click monitor — dismiss bar when clicking anywhere outside it
        self._click_monitor = None
        self._install_click_monitor()

        # Status view (recording/processing/result)
        self.view = OverlayView.alloc().initWithFrame_(
            AppKit.NSMakeRect(0, 0, BAR_WIDTH, BAR_HEIGHT)
        )
        self.window.setContentView_(self.view)

        # Text field (hidden until typing mode)
        self._text_field = AppKit.NSTextField.alloc().initWithFrame_(
            AppKit.NSMakeRect(32, (BAR_HEIGHT - 22) / 2, BAR_WIDTH - 44, 22)
        )
        self._text_field.setFont_(AppKit.NSFont.systemFontOfSize_(14))
        self._text_field.setTextColor_(AppKit.NSColor.whiteColor())
        self._text_field.setBackgroundColor_(AppKit.NSColor.clearColor())
        self._text_field.setBezeled_(False)
        self._text_field.setFocusRingType_(AppKit.NSFocusRingTypeNone)
        self._text_field.setPlaceholderString_("Type a command and press Enter...")
        self._text_field.setHidden_(True)

        # Set up delegate for Enter key
        self._text_delegate = TextFieldDelegate.alloc().init()
        self._text_delegate.controller = self
        self._text_field.setDelegate_(self._text_delegate)

        self.view.addSubview_(self._text_field)

    @objc.python_method
    def _install_click_monitor(self):
        """Install a global event monitor that dismisses the bar on any click outside it."""
        if self._click_monitor:
            return

        mask = (
            AppKit.NSEventMaskLeftMouseDown
            | AppKit.NSEventMaskRightMouseDown
            | AppKit.NSEventMaskOtherMouseDown
        )

        def handler(event):
            if not self._visible:
                return event
            # Check if click is inside the overlay window
            click_loc = AppKit.NSEvent.mouseLocation()
            win_frame = self.window.frame()
            if not AppKit.NSPointInRect(click_loc, win_frame):
                # Click outside — dismiss and notify parent
                self._hide()
                sys.stdout.write("dismissed\n")
                sys.stdout.flush()
            return event

        self._click_monitor = AppKit.NSEvent.addGlobalMonitorForEventsMatchingMask_handler_(
            mask, handler
        )

    @objc.python_method
    def _show(self, state, text=None):
        info = STATES.get(state, STATES["recording"])
        r, g, b, a = info["accent"]
        color = AppKit.NSColor.colorWithCalibratedRed_green_blue_alpha_(r, g, b, a)
        display_text = text if text else info["text"]

        self.view.setAccentColor_(color)
        self.view.setText_(display_text)

        # Exit typing mode for non-typing states
        if state != "typing" and self._typing_mode:
            self._exit_typing_mode()

        if not self._visible:
            self.window.orderFront_(None)
            self._visible = True

        if self._hide_timer and self._hide_timer.isValid():
            self._hide_timer.invalidate()
            self._hide_timer = None

    @objc.python_method
    def _hide(self):
        if self._typing_mode:
            self._exit_typing_mode()
        if self._visible:
            self.window.orderOut_(None)
            self._visible = False

    @objc.python_method
    def _enter_typing_mode(self):
        """Show text input field, allow clicks, focus the field."""
        self._typing_mode = True
        self.view.setText_("")  # Clear the status text so it doesn't overlap
        self._text_field.setStringValue_("")
        self._text_field.setHidden_(False)
        self.window.setIgnoresMouseEvents_(False)  # Allow clicks

        # Activate app and grab focus — needs to happen in this order
        app = AppKit.NSApplication.sharedApplication()
        app.activateIgnoringOtherApps_(True)
        self.window.makeKeyAndOrderFront_(None)
        self.window.makeFirstResponder_(self._text_field)

        # Re-grab focus after a tiny delay (macOS sometimes needs this)
        NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
            0.1, self, self.grabFocus_, None, False
        )

    def grabFocus_(self, timer):
        """Delayed focus grab to ensure text field is ready."""
        AppKit.NSApplication.sharedApplication().activateIgnoringOtherApps_(True)
        self.window.makeKeyAndOrderFront_(None)
        self.window.makeFirstResponder_(self._text_field)

    @objc.python_method
    def _exit_typing_mode(self):
        """Hide text field, go back to click-through mode."""
        self._typing_mode = False
        self._text_field.setHidden_(True)
        self.window.setIgnoresMouseEvents_(True)

    @objc.python_method
    def _enter_processing_after_type(self):
        """Called after user submits typed text."""
        self._exit_typing_mode()
        self._show("processing")

    def autoHide_(self, timer):
        self._hide()

    def handleCommand_(self, cmd):
        """Called from main thread via performSelectorOnMainThread."""
        if cmd == "quit":
            AppKit.NSApplication.sharedApplication().terminate_(None)
        elif cmd == "hide":
            self._hide()
        elif cmd == "recording":
            self._show("recording")
        elif cmd == "processing":
            self._show("processing")
        elif cmd == "typing":
            self._show("typing")
            self._enter_typing_mode()
        elif cmd.startswith("result:"):
            text = cmd[7:].strip()
            if len(text) > 50:
                text = text[:47] + "..."
            self._show("result", text)
            self._hide_timer = NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
                FADE_AFTER_SEC, self, self.autoHide_, None, False
            )

    def startStdinReader(self):
        def reader():
            for line in sys.stdin:
                cmd = line.strip()
                if not cmd:
                    continue
                self.performSelectorOnMainThread_withObject_waitUntilDone_(
                    self.handleCommand_, cmd, False
                )
        t = threading.Thread(target=reader, daemon=True)
        t.start()


def main():
    controller = OverlayController.alloc().init()
    controller.setup()

    # Quick test mode
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        def run_test():
            import time
            cmds = [
                (0.1, "recording"),
                (2.0, "processing"),
                (4.0, "result:Logged chicken breast to Lunch."),
                (7.0, "typing"),
                (15.0, "quit"),
            ]
            for delay, cmd in cmds:
                time.sleep(delay)
                controller.performSelectorOnMainThread_withObject_waitUntilDone_(
                    controller.handleCommand_, cmd, False
                )
        t = threading.Thread(target=run_test, daemon=True)
        t.start()
    else:
        controller.startStdinReader()

    AppKit.NSApplication.sharedApplication().run()


if __name__ == "__main__":
    main()
