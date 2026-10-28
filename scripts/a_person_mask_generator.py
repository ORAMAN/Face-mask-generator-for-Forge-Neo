from functools import reduce
import cv2
import gradio as gr
import os
import re
from PIL import Image
import numpy as np
import mediapipe as mp

import modules.scripts as scripts
from modules.paths_internal import models_path
from modules.ui_components import FormRow, FormGroup

BaseOptions = mp.tasks.BaseOptions
ImageSegmenter = mp.tasks.vision.ImageSegmenter
ImageSegmenterOptions = mp.tasks.vision.ImageSegmenterOptions
VisionRunningMode = mp.tasks.vision.RunningMode

MASK_OPTION_0_BACKGROUND = 'background'
MASK_OPTION_1_HAIR = 'hair'
MASK_OPTION_2_BODY = 'body (skin)'
MASK_OPTION_3_FACE = 'face (skin)'
MASK_OPTION_4_CLOTHES = 'clothes'

title = "A Person Mask Generator"


class Script(scripts.Script):

    def __init__(self):
        self.img2img: gr.Image = None

        # ForgeCanvas -> LogicalImage mapping
        self._logical_images_by_id = {}
        self.img2img_canvas_uuid = None
        self.img2img_image_background = None

    def title(self):
        return title

    def show(self, is_img2img):
        return scripts.AlwaysVisible
    
    def get_mediapipe_image(self, image: Image) -> mp.Image:
        numpy_image = np.asarray(image)

        # Grayscale -> RGB
        if numpy_image.ndim == 2:
            numpy_image = cv2.cvtColor(
                numpy_image,
                cv2.COLOR_GRAY2RGB
            )
            image_format = mp.ImageFormat.SRGB

        # RGB/RGBA
        elif numpy_image.ndim == 3:
            if numpy_image.shape[2] == 4:
                image_format = mp.ImageFormat.SRGBA
            elif numpy_image.shape[2] == 3:
                image_format = mp.ImageFormat.SRGB
            else:
                raise ValueError(
                    f"Unsupported channel count: {numpy_image.shape}"
                )

        else:
            raise ValueError(
                f"Unsupported image shape: {numpy_image.shape}"
            )

        return mp.Image(image_format=image_format, data=numpy_image)



    def generate_mask(
            self,
            image: Image,
            mask_targets: list[str],
            mask_dilation: int
    ) -> Image:

        if image is None or len(mask_targets) == 0:
            return None

        model_folder_name = 'mediapipe'
        model_file_name = 'selfie_multiclass_256x256.tflite'

        model_folder_path = (
            os.path.join(models_path, model_folder_name)
            if not models_path.endswith(model_folder_name)
            else models_path
        )

        os.makedirs(model_folder_path, exist_ok=True)

        model_path = os.path.join(
            model_folder_path,
            model_file_name
        )

        model_url = (
            'https://storage.googleapis.com/mediapipe-models/'
            'image_segmenter/selfie_multiclass_256x256/float32/latest/'
            'selfie_multiclass_256x256.tflite'
        )

        if not os.path.exists(model_path):
            import urllib.request

            print(
                "[A Person Mask] Downloading "
                "'selfie_multiclass_256x256.tflite' model"
            )

            urllib.request.urlretrieve(
                model_url,
                model_path
            )

        options = ImageSegmenterOptions(
            base_options=BaseOptions(
                model_asset_path=model_path
            ),
            running_mode=VisionRunningMode.IMAGE,
            output_category_mask=False,
            output_confidence_masks=True
        )

        with ImageSegmenter.create_from_options(options) as segmenter:

            media_pipe_image = self.get_mediapipe_image(
                image=image
            )

            segmented_masks = segmenter.segment(
                media_pipe_image
            )

            # MediaPipe multiclass model:
            # 0 - background
            # 1 - hair
            # 2 - body / skin
            # 3 - face / skin
            # 4 - clothes
            # 5 - others / accessories

            mask_indices = {
                MASK_OPTION_0_BACKGROUND: 0,
                MASK_OPTION_1_HAIR: 1,
                MASK_OPTION_2_BODY: 2,
                MASK_OPTION_3_FACE: 3,
                MASK_OPTION_4_CLOTHES: 4,
            }

            masks = []

            for target in mask_targets:
                mask_index = mask_indices.get(target)

                if mask_index is None:
                    continue

                masks.append(
                    segmented_masks.confidence_masks[mask_index]
                )

            if not masks:
                return None

            image_data = media_pipe_image.numpy_view()
            image_shape = image_data.shape

            target_height = image_shape[0]
            target_width = image_shape[1]

            mask_arrays = []

            for mask in masks:

                mask_data = np.asarray(
                    mask.numpy_view()
                )

                mask_data = np.squeeze(mask_data)

                if mask_data.ndim != 2:
                    raise ValueError(
                        "Unexpected MediaPipe mask shape: "
                        f"{mask_data.shape}"
                    )

                if (
                    mask_data.shape[0] != target_height
                    or mask_data.shape[1] != target_width
                ):
                    mask_data = cv2.resize(
                        mask_data,
                        (target_width, target_height),
                        interpolation=cv2.INTER_LINEAR
                    )

                mask_binary = (
                    mask_data > 0.25
                ).astype(np.uint8) * 255

                mask_arrays.append(mask_binary)

            # Merge selected categories.
            merged_mask = reduce(
                np.maximum,
                mask_arrays
            )

            # Positive = dilate, negative = erode.
            if mask_dilation > 0:

                kernel = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE,
                    (
                        2 * mask_dilation + 1,
                        2 * mask_dilation + 1
                    )
                )

                merged_mask = cv2.dilate(
                    merged_mask,
                    kernel
                )

            elif mask_dilation < 0:

                dilation = abs(mask_dilation)

                kernel = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE,
                    (
                        2 * dilation + 1,
                        2 * dilation + 1
                    )
                )

                merged_mask = cv2.erode(
                    merged_mask,
                    kernel
                )

            # Create RGBA mask.
            #
            # RGB:
            #   black = unmasked
            #   white = masked
            #
            # Alpha:
            #   255 everywhere
            mask_rgba = np.zeros(
                (
                    target_height,
                    target_width,
                    4
                ),
                dtype=np.uint8
            )

            mask_rgba[:, :, 3] = 255

            mask_rgba[merged_mask > 0, 0:3] = 255

            return Image.fromarray(
                mask_rgba,
                mode="RGBA"
            )

    # How the script's is displayed in the UI. See https://gradio.app/docs/#components
    # for the different UI components you can use and how to create them.
    # Most UI components can return a value, such as a boolean for a checkbox.
    # The returned values are passed to the run method as parameters.
    def ui(self, is_img2img):
        if is_img2img:
            with gr.Accordion(title, open=False, elem_id="a_person_mask_generator_accordion") as accordion:
                with gr.Blocks():
                    with gr.Row():
                        with gr.Column():
                            enabled = gr.Checkbox(
                                label="Enable",
                                value=False,
                                elem_id=f"a_person_mask_generator_enable_checkbox",
                            )


                            mask_targets = gr.Dropdown(
                                label="Mask",
                                multiselect=True,
                                choices=[
                                    MASK_OPTION_0_BACKGROUND,
                                    MASK_OPTION_1_HAIR,
                                    MASK_OPTION_2_BODY,
                                    MASK_OPTION_3_FACE,
                                    MASK_OPTION_4_CLOTHES,
                                ],
                                value=[MASK_OPTION_3_FACE],
                                elem_id="a_person_mask_generator_mask_dropdown",
                                interactive=True,
                            )

                            gr.HTML("<div style='margin: 8px 0px !important;'></div>")

                            preview_button = gr.Button(value="Preview Mask *", elem_id="person_mask_generator_preview_button")
                            gr.HTML("<div style='margin: 8px 0px !important; opacity: 0.75;'>* Not available on 'Inpaint' tab.</div>")

                            inpaint_dilation = gr.Slider(
                                label='Mask dilation / erosion, pixels',
                                minimum=-64,
                                maximum=64,
                                step=1,
                                value=0,
                                elem_id="a_person_mask_generator_inpaint_dilation"
                            )

                            override_inpaint_enabled = gr.Checkbox(
                                label="Override mask settings",
                                value=False,
                                elem_id="a_person_mask_generator_override_inpaint_checkbox",
                            )

                            with FormGroup(elem_id="a_person_mask_generator_inpaint_controls", visible=False) as a_person_mask_generator_inpaint_controls:
                                with FormRow():
                                    mask_blur = gr.Slider(label='Mask blur', minimum=0, maximum=64, step=1, value=4, elem_id="a_person_mask_generator_mask_blur")
                                with FormRow():
                                    inpainting_mask_invert = gr.Radio(label='Mask mode', choices=['Inpaint masked', 'Inpaint not masked'], value='Inpaint masked', type="index",
                                                                        elem_id="a_person_mask_generator_mask_mode")
                                with FormRow():
                                    inpainting_fill = gr.Radio(label='Masked content', choices=['fill', 'original', 'latent noise', 'latent nothing'], value='original',
                                                                type="index",
                                                                elem_id="a_person_mask_generator_inpainting_fill")
                                with FormRow():
                                    inpaint_full_res = gr.Radio(label="Inpaint area", choices=["Whole picture", "Only masked"], type="index", value="Whole picture",
                                                                elem_id="a_person_mask_generator_inpaint_full_res")

                                with FormRow():
                                    inpaint_full_res_padding = gr.Slider(label='Only masked padding, pixels', minimum=0, maximum=256, step=4, value=32,
                                                                        elem_id="a_person_mask_generator_inpaint_full_res_padding")

                        with gr.Column():
                            preview_mask_image = gr.Image(value=None, label="Mask Preview", show_label=True, interactive=False, height=256, width=256,
                                                        elem_id="a_person_mask_generator_preview_mask_image")

                def update_preview_image(
                    image,
                    mask_targets: list[str],
                    inpaint_dilation: float
                ):
                    try:
                        if image is None:
                            return None

                        if not mask_targets:
                            return None

                        return self.generate_mask(
                            image=image,
                            mask_targets=mask_targets,
                            mask_dilation=int(inpaint_dilation)
                        )

                    except Exception as e:
                        import traceback

                        print(
                            f"[A Person Mask] Preview error: "
                            f"{type(e).__name__}: {e}"
                        )

                        traceback.print_exc()

                        return None                             

                # change handlers
                mask_targets.change(
                    fn=update_preview_image,
                    inputs=[
                        self.img2img_image_background,
                        mask_targets,
                        inpaint_dilation
                    ],
                    outputs=preview_mask_image
                )

                enabled.change(
                    fn=update_preview_image,
                    inputs=[
                        self.img2img_image_background,
                        mask_targets,
                        inpaint_dilation
                    ],
                    outputs=preview_mask_image
                )

                preview_button.click(
                    fn=update_preview_image,
                    inputs=[
                        self.img2img_image_background,
                        mask_targets,
                        inpaint_dilation
                    ],
                    outputs=preview_mask_image
                )

                inpaint_dilation.release(
                    fn=update_preview_image,
                    inputs=[
                        self.img2img_image_background,
                        mask_targets,
                        inpaint_dilation
                    ],
                    outputs=preview_mask_image
                )

                def toggle_inpaint_controls(show: bool):
                    return {
                        a_person_mask_generator_inpaint_controls: gr.update(visible=show)
                    }

                override_inpaint_enabled.change(
                    fn=toggle_inpaint_controls,
                    inputs=[override_inpaint_enabled],
                    outputs=a_person_mask_generator_inpaint_controls
                )

                return [
                    enabled,
                    mask_targets,
                    override_inpaint_enabled,
                    mask_blur,
                    inpainting_mask_invert,
                    inpainting_fill,
                    inpaint_full_res,
                    inpaint_full_res_padding,
                    inpaint_dilation,
                ]
        else:
            return ()

    # From: https://github.com/LonicaMewinsky/gif2gif/blob/main/scripts/gif2gif.py
    # Grab the img2img image components for update later
    # # Maybe there's a better way to do this?

    def after_component(self, component, **kwargs):

        classes = getattr(component, "elem_classes", None) or []
        elem_id = getattr(component, "elem_id", None)

        # ForgeCanvas LogicalImage background
        if "logical_image_background" in classes:

            self._logical_images_by_id[elem_id] = component

            if elem_id == self.img2img_canvas_uuid:
                self.img2img_image_background = component

            return component

        # ForgeCanvas HTML: img2img_image
        if elem_id == "img2img_image":

            self.img2img_image = component

            html = str(
                getattr(component, "value", None) or ""
            )

            match = re.search(
                r'id=["\']container_(uuid_[0-9a-f]+)["\']',
                html
            )

            if match:

                self.img2img_canvas_uuid = match.group(1)

                background = self._logical_images_by_id.get(
                    self.img2img_canvas_uuid
                )

                if background is not None:
                    background_classes = (
                        getattr(background, "elem_classes", None)
                        or []
                    )

                    if "logical_image_background" in background_classes:
                        self.img2img_image_background = background

            return component

        return component

    """
    This function is called very early during processing begins for AlwaysVisible scripts.
    You can modify the processing object (p) here, inject hooks, etc.
    args contains all values returned by components from ui()
    """

    def before_process(
            self,
            p,
            enabled: bool = False,
            mask_targets: list[str] | None = None,
            override_inpaint_enabled: bool = False,
            mask_blur: int = 0,
            inpainting_mask_invert: int = 0,
            inpainting_fill: int = 0,
            inpaint_full_res: bool = False,
            inpaint_full_res_padding: int = 0,
            inpaint_dilation: int = 0,
    ):
        if mask_targets is None:
            mask_targets = []

        if enabled and len(p.init_images) > 0 and len(mask_targets) > 0:
            p.image_mask = self.generate_mask(
                image=p.init_images[-1],
                mask_targets=mask_targets,
                mask_dilation=inpaint_dilation
            )

            if override_inpaint_enabled:
                p.mask_blur = mask_blur
                p.inpainting_mask_invert = inpainting_mask_invert
                p.inpainting_fill = inpainting_fill
                p.inpaint_full_res = inpaint_full_res
                p.inpaint_full_res_padding = inpaint_full_res_padding
