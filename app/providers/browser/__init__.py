"""Browser adapters.

Both modules import their heavy dependency lazily. Playwright and browser_use
are optional, and importing them at module scope would make an optional
integration a hard startup requirement.
"""

from app.providers.browser.browser_use import BrowserUseProvider
from app.providers.browser.unavailable import UnavailableBrowserProvider

__all__ = ["BrowserUseProvider", "UnavailableBrowserProvider"]
