"""Survive viam-server reconfigure when a dependency (for example a camera) restarts.

viam-sdk 0.83 rebuilds a modular resource by deleting it and constructing a new
one. ``ResourceManager.remove_resource`` raises ``KeyError`` when that name is
not in the manager. The module's robot client does this during ``refresh``:
``ResourceNames`` still lists a service from the previous pass (the other
stereo segmenter, while it is mid-reconfigure) that this process never
registered a client for. The ``KeyError`` fails the ``Reconfigure`` RPC after
the local resource has already been removed, and viam-server leaves it failed
until the module process is restarted.

Installing this patch makes removal of an unknown name a no-op, and keeps a
resource that implements ``reconfigure`` in place so the SAM3 weights stay
loaded and the new dependency clients are swapped in.
"""

from __future__ import annotations

from inspect import isawaitable, iscoroutinefunction

from viam.errors import ResourceNotFoundError
from viam.logging import getLogger, update_log_level
from viam.module.module import Module
from viam.module.types import Stoppable
from viam.proto.common import ResourceName
from viam.proto.module import AddResourceRequest
from viam.resource.base import ResourceBase
from viam.resource.manager import ResourceManager
from viam.resource.types import API

LOGGER = getLogger(__name__)


async def _safe_remove_resource(self: ResourceManager, name: ResourceName) -> None:
    """Close and drop ``name``. A missing name is already gone."""
    with self._lock:
        try:
            resource = self.resources[name]
        except KeyError:
            LOGGER.debug(
                "Ignoring remove of %s; it is not registered in this process",
                name,
            )
            return
        try:
            await resource.close()
        finally:
            # One object can be stored under several ResourceNames (class MRO).
            # Drop every alias so a later register does not see a duplicate.
            for key, value in list(self.resources.items()):
                if key == name or value is resource:
                    self.resources.pop(key, None)


def _apply_log_level(resource: ResourceBase, config) -> None:
    level = getattr(getattr(config, "log_configuration", None), "level", "") or ""
    if not level or not hasattr(resource, "logger"):
        return
    update_log_level(resource.logger, level.upper())


async def _reconfigure_resource(self: Module, request) -> None:
    """Reconfigure without deleting the resource when it can update in place."""
    config = request.config
    api = API.from_string(config.api)
    name = config.name
    rn = ResourceName(
        namespace=api.namespace,
        type=api.resource_type,
        subtype=api.resource_subtype,
        name=name,
    )
    try:
        resource = self.server.get_resource(ResourceBase, rn)
    except ResourceNotFoundError:
        # A previous reconfigure removed the resource and then failed. Add it
        # back so this process recovers without a module restart.
        resource = None

    if resource is not None and callable(getattr(resource, "reconfigure", None)):
        dependencies = await self._get_dependencies(request.dependencies)
        outcome = resource.reconfigure(config, dependencies)
        if isawaitable(outcome):
            await outcome
        _apply_log_level(resource, config)
        return

    if resource is not None:
        if isinstance(resource, Stoppable):
            if iscoroutinefunction(resource.stop):
                await resource.stop()
            else:
                resource.stop()
        await self.server.remove_resource(rn)
    await self.add_resource(AddResourceRequest(config=request.config, dependencies=request.dependencies))


def install() -> None:
    """Patch the viam SDK process-wide. Safe to call more than once."""
    if getattr(ResourceManager.remove_resource, "_sam3_safe_remove", False):
        return
    _safe_remove_resource._sam3_safe_remove = True  # type: ignore[attr-defined]
    ResourceManager.remove_resource = _safe_remove_resource  # type: ignore[method-assign]
    Module.reconfigure_resource = _reconfigure_resource  # type: ignore[method-assign]
