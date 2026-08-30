"""FastAPI surface for incremental pet enrollment and identification."""

from __future__ import annotations

import secrets
from typing import Annotated

from fastapi import (
    APIRouter,
    Depends,
    FastAPI,
    File,
    Form,
    Header,
    HTTPException,
    Query,
    UploadFile,
)
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from .gallery_service import (
    GalleryServiceError,
    GalleryNotFound,
    PetIdentificationService,
    UploadPayload,
)


class PetUpdateRequest(BaseModel):
    display_name: str = Field(min_length=1, max_length=128)


class HistoryReviewRequest(BaseModel):
    status: str = Field(pattern="^(unreviewed|correct|incorrect|uncertain)$")
    note: str | None = Field(default=None, max_length=1000)


def create_app(
    service: PetIdentificationService,
    *,
    api_key: str | None = None,
) -> FastAPI:
    """Create an application around one already initialized GPU service."""

    app = FastAPI(
        title="Pet ReID API",
        version="1.0.0",
        description=(
            "Enroll pet reference images and identify query images with a locked "
            "model-bound gallery. OpenAPI documentation is available at /docs."
        ),
    )
    app.state.pet_service = service

    @app.exception_handler(GalleryServiceError)
    async def gallery_error_handler(_request, error: GalleryServiceError):
        return JSONResponse(
            status_code=error.status_code,
            content={
                "error": {
                    "code": error.code,
                    "message": str(error),
                    "details": error.details,
                }
            },
        )

    def authorize(
        x_api_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
    ) -> None:
        if api_key is not None and (
            x_api_key is None or not secrets.compare_digest(x_api_key, api_key)
        ):
            raise HTTPException(
                status_code=401,
                detail="a valid X-API-Key header is required",
                headers={"WWW-Authenticate": "ApiKey"},
            )

    protected = APIRouter(prefix="/v1", dependencies=[Depends(authorize)])

    async def read_upload(upload: UploadFile) -> UploadPayload:
        try:
            data = await upload.read(service.maximum_upload_bytes + 1)
            return UploadPayload(
                filename=upload.filename or "upload",
                content_type=upload.content_type,
                data=data,
            )
        finally:
            await upload.close()

    def not_found(kind: str, identifier: str) -> GalleryNotFound:
        return GalleryNotFound(f"{kind} {identifier!r} was not found")

    @app.get("/health", tags=["system"])
    def health():
        return service.health()

    @protected.get("/pets", tags=["gallery"])
    def list_pets():
        pets = service.store.list_pets()
        return {"pets": pets, "count": len(pets)}

    @protected.get("/pets/{pet_id}", tags=["gallery"])
    def get_pet(pet_id: str):
        return service.store.get_pet(pet_id)

    @protected.patch("/pets/{pet_id}", tags=["gallery"])
    @protected.put("/pets/{pet_id}", tags=["gallery"], include_in_schema=False)
    def update_pet(pet_id: str, request: PetUpdateRequest):
        return service.store.update_pet(pet_id, request.display_name)

    @protected.post("/pets/{pet_id}/images", status_code=201, tags=["gallery"])
    async def enroll_images(
        pet_id: str,
        files: Annotated[list[UploadFile], File(description="One or more pet images")],
        display_name: Annotated[str | None, Form()] = None,
    ):
        payloads = [await read_upload(upload) for upload in files]
        return await run_in_threadpool(
            service.enroll,
            pet_id,
            payloads,
            display_name=display_name,
        )

    @protected.get("/pets/{pet_id}/images/{image_id}", tags=["gallery"])
    def get_image(pet_id: str, image_id: str):
        path, content_type, original_filename = service.store.image_path(
            pet_id, image_id
        )
        return FileResponse(path, media_type=content_type, filename=original_filename)

    @protected.delete("/pets/{pet_id}/images/{image_id}", tags=["gallery"])
    def delete_image(pet_id: str, image_id: str):
        return service.store.delete_image(pet_id, image_id)

    @protected.delete("/pets/{pet_id}", tags=["gallery"])
    def delete_pet(pet_id: str):
        return service.store.delete_pet(pet_id)

    @protected.post("/identify", tags=["identification"])
    async def identify(
        file: Annotated[UploadFile, File(description="Query pet image")],
        top_k: Annotated[int, Query(ge=1, le=50)] = 5,
        match_threshold: Annotated[float | None, Query(ge=-1.0, le=1.0)] = None,
        minimum_margin: Annotated[float | None, Query(ge=0.0, le=2.0)] = None,
    ):
        payload = await read_upload(file)
        return await run_in_threadpool(
            service.identify,
            payload,
            top_k=top_k,
            match_threshold=match_threshold,
            minimum_margin=minimum_margin,
        )

    @protected.get("/history", tags=["history"])
    def list_history(
        page: Annotated[int, Query(ge=1)] = 1,
        page_size: Annotated[int, Query(ge=1, le=200)] = 25,
        source: str | None = None,
        accepted: bool | None = None,
        review_status: str | None = None,
        pet_id: str | None = None,
        hard_only: bool = False,
    ):
        try:
            return service.operations.list_history(
                page=page,
                page_size=page_size,
                source=source,
                accepted=accepted,
                review_status=review_status,
                pet_id=pet_id,
                hard_only=hard_only,
            )
        except ValueError as error:
            from .gallery_service import InvalidGalleryRequest

            raise InvalidGalleryRequest(str(error)) from error

    @protected.get("/history/{history_id}", tags=["history"])
    def get_history(history_id: str):
        try:
            return service.operations.get_history(history_id)
        except KeyError as error:
            raise not_found("history", history_id) from error

    @protected.get("/history/{history_id}/image", tags=["history"])
    def get_history_image(history_id: str):
        try:
            path, content_type, filename = service.operations.history_image(history_id)
        except KeyError as error:
            raise not_found("history image", history_id) from error
        return FileResponse(path, media_type=content_type, filename=filename)

    @protected.patch("/history/{history_id}/review", tags=["history"])
    @protected.put(
        "/history/{history_id}/review",
        tags=["history"],
        include_in_schema=False,
    )
    def review_history(history_id: str, request: HistoryReviewRequest):
        try:
            return service.operations.review_history(
                history_id,
                status=request.status,
                note=request.note,
            )
        except KeyError as error:
            raise not_found("history", history_id) from error
        except ValueError as error:
            from .gallery_service import InvalidGalleryRequest

            raise InvalidGalleryRequest(str(error)) from error

    @protected.delete("/history/{history_id}", tags=["history"])
    def delete_history(history_id: str):
        try:
            return service.operations.delete_history(history_id)
        except KeyError as error:
            raise not_found("history", history_id) from error

    @protected.post("/batches", status_code=202, tags=["batch"])
    async def create_batch(
        files: Annotated[list[UploadFile], File(description="Batch query images")],
        name: Annotated[str, Form(max_length=128)] = "批量测试",
        expected_pet_ids: Annotated[list[str] | None, Form()] = None,
        top_k: Annotated[int, Query(ge=1, le=50)] = 5,
        match_threshold: Annotated[float | None, Query(ge=-1.0, le=1.0)] = None,
        minimum_margin: Annotated[float | None, Query(ge=0.0, le=2.0)] = None,
    ):
        payloads = [await read_upload(upload) for upload in files]
        labels = expected_pet_ids
        if labels is not None:
            labels = [label.strip() or None for label in labels]
        return service.create_batch(
            name=name,
            uploads=payloads,
            expected_pet_ids=labels,
            top_k=top_k,
            match_threshold=match_threshold,
            minimum_margin=minimum_margin,
        )

    @protected.get("/batches", tags=["batch"])
    def list_batches(
        page: Annotated[int, Query(ge=1)] = 1,
        page_size: Annotated[int, Query(ge=1, le=100)] = 20,
    ):
        return service.operations.list_batches(page=page, page_size=page_size)

    @protected.get("/batches/{batch_id}", tags=["batch"])
    def get_batch(batch_id: str):
        try:
            return service.operations.get_batch(batch_id)
        except KeyError as error:
            raise not_found("batch", batch_id) from error

    @protected.delete("/batches/{batch_id}", tags=["batch"])
    def cancel_batch(batch_id: str):
        try:
            return service.cancel_batch(batch_id)
        except KeyError as error:
            raise not_found("batch", batch_id) from error

    @protected.get("/batches/{batch_id}/results.csv", tags=["batch"])
    def batch_csv(batch_id: str):
        try:
            content = service.operations.batch_csv(batch_id)
        except KeyError as error:
            raise not_found("batch", batch_id) from error
        return Response(
            content=content.encode("utf-8-sig"),
            media_type="text/csv; charset=utf-8",
            headers={
                "Content-Disposition": f'attachment; filename="batch-{batch_id}.csv"'
            },
        )

    @protected.get("/hard-cases", tags=["administration"])
    def hard_cases(
        page: Annotated[int, Query(ge=1)] = 1,
        page_size: Annotated[int, Query(ge=1, le=200)] = 50,
        review_status: str | None = None,
    ):
        try:
            return service.operations.list_history(
                page=page,
                page_size=page_size,
                hard_only=True,
                review_status=review_status,
            )
        except ValueError as error:
            from .gallery_service import InvalidGalleryRequest

            raise InvalidGalleryRequest(str(error)) from error

    @protected.get("/gallery/backup", tags=["administration"])
    def backup_gallery():
        filename, content = service.create_gallery_backup()
        return Response(
            content=content,
            media_type="application/zip",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"'
            },
        )

    @protected.post("/gallery/restore", tags=["administration"])
    async def restore_gallery(
        file: Annotated[UploadFile, File(description="Gallery backup ZIP")],
    ):
        try:
            data = await file.read(service.maximum_backup_bytes + 1)
        finally:
            await file.close()
        return await run_in_threadpool(service.restore_gallery_backup, data)

    app.include_router(protected)
    return app
