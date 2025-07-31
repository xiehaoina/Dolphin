import json
from PIL import Image
from loguru import logger

from src.models.chat.chat_model import ChatModel
from src.pipelines.processor.processor import Processor
from src.utils.image_process.load_image import encode_image_base64


class VLMMergeProcessor(Processor):
    def __init__(self, model: ChatModel, prompt_path: str = "src/prompts/merge_json.md"):
        """
        Initializes the VLMMergeProcessor.
        """
        self.model = model
        try:
            with open(prompt_path, 'r', encoding='utf-8') as f:
                self.prompt_template = f.read()
        except FileNotFoundError:
            logger.error(f"Prompt template file not found at: {prompt_path}")
            raise
        except Exception as e:
            logger.error(f"Error reading prompt template file: {e}")
            raise

    def _log_dict_diff(self, old_data: dict, new_data: dict):
        old_keys = set(old_data.keys())
        new_keys = set(new_data.keys())

        added_keys = new_keys - old_keys
        if added_keys:
            logger.debug(f"  Added keys: {', '.join(sorted(list(added_keys)))}")

        removed_keys = old_keys - new_keys
        if removed_keys:
            logger.debug(f"  Removed keys: {', '.join(sorted(list(removed_keys)))}")

        common_keys = old_keys.intersection(new_keys)
        for key in sorted(list(common_keys)):
            if old_data[key] != new_data[key]:
                old_val = json.dumps(old_data[key], ensure_ascii=False)
                new_val = json.dumps(new_data[key], ensure_ascii=False)
                logger.debug(f"  Modified key '{key}':")
                logger.debug(f"    - Original: {old_val}")
                logger.debug(f"    + Refined:  {new_val}")

    def _log_list_diff(self, old_list: list, new_list: list):
        if len(old_list) != len(new_list):
            logger.debug(f"List length changed from {len(old_list)} to {len(new_list)}.")

        try:
            # Attempt a keyed diff if items appear to be dicts with a unique 'id'
            old_items_by_id = {item['id']: item for item in old_list if isinstance(item, dict) and 'id' in item}
            new_items_by_id = {item['id']: item for item in new_list if isinstance(item, dict) and 'id' in item}
            if len(old_items_by_id) == len(old_list) and len(new_items_by_id) == len(new_list):
                old_ids, new_ids = set(old_items_by_id.keys()), set(new_items_by_id.keys())
                added, removed, common = new_ids - old_ids, old_ids - new_ids, old_ids & new_ids

                if added:
                    logger.debug("--- Added list items ---")
                    for i in sorted(list(added)): logger.debug(json.dumps(new_items_by_id[i], ensure_ascii=False, indent=2))
                if removed:
                    logger.debug("--- Removed list items ---")
                    for i in sorted(list(removed)): logger.debug(json.dumps(old_items_by_id[i], ensure_ascii=False, indent=2))
                
                modified_items = [i for i in common if old_items_by_id[i] != new_items_by_id[i]]
                if modified_items:
                    logger.debug("--- Modified list items ---")
                    for i in sorted(modified_items):
                        logger.debug(f"Item with id '{i}' was modified:")
                        self._log_dict_diff(old_items_by_id[i], new_items_by_id[i])
                return
        except (TypeError, KeyError):
            pass  # Fallback to content-based diff

        logger.debug("List items have no consistent 'id' key, performing content-based diff.")
        old_set = {json.dumps(i, sort_keys=True) for i in old_list}
        new_set = {json.dumps(i, sort_keys=True) for i in new_list}
        
        removed_items = old_set - new_set
        if removed_items:
            logger.debug(f"--- Removed/Modified items ({len(removed_items)}) ---")
            for item in removed_items: logger.debug(json.dumps(json.loads(item), ensure_ascii=False, indent=2))
        
        added_items = new_set - old_set
        if added_items:
            logger.debug(f"--- Added/Modified items ({len(added_items)}) ---")
            for item in added_items: logger.debug(json.dumps(json.loads(item), ensure_ascii=False, indent=2))

    def _log_json_diff(self, old_json_str: str, new_json_str: str):
        try:
            old_data = json.loads(old_json_str)
            new_data = json.loads(new_json_str)

            if old_data == new_data:
                logger.debug("VLM refinement resulted in no changes to the JSON.")
                return

            logger.debug("--- JSON Diff (Original vs. Refined) ---")
            if isinstance(old_data, dict) and isinstance(new_data, dict):
                self._log_dict_diff(old_data, new_data)
            elif isinstance(old_data, list) and isinstance(new_data, list):
                self._log_list_diff(old_data, new_data)
            else:
                logger.debug(f"JSON types differ: from {type(old_data).__name__} to {type(new_data).__name__}")
            logger.debug("-----------------------------------------")

        except json.JSONDecodeError:
            logger.warning("Could not compare JSONs because the refined JSON is invalid.")
        except Exception as e:
            logger.error(f"An unexpected error occurred during JSON diff: {e}", exc_info=True)

    def process(self, input_data: dict, image: Image.Image) -> dict:
        """
        Refines the initial JSON output by using a VLM to merge and correct information.
        """
        try:
            input_json_str = json.dumps(input_data, ensure_ascii=False, indent=2)

            image_tag = f"[Base64 Image: {len(encode_image_base64(image))} bytes]"
            prompt = self.prompt_template.replace("{{INPUT_JSON}}", input_json_str)
            prompt = prompt.replace("{{IMAGE_DOCUMENT}}", image_tag)
            print(prompt)
            resp_str = self.model.inference(prompt, image)

            # Log the difference between the original and refined JSON
            # self._log_json_diff(input_json_str, cleaned_resp_str)
            # Parse the refined JSON from the model's response
            refined_output = json.loads(resp_str)
            print(refined_output)
            return refined_output

        except json.JSONDecodeError:
            logger.error(f"Failed to decode JSON from VLM response. Raw response: {resp_str}")
            return input_data  # Fallback to original data
        except Exception as e:
            logger.error(f"An error occurred in VLMMergeProcessor: {e}")
            return input_data  # Fallback to original data

    def batch_process(self, *args, **kwargs):
        raise NotImplementedError("Batch processing is not implemented for this processor.")
    
    