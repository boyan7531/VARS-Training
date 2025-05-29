import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.io import read_video, read_video_timestamps
import json
from pathlib import Path
import random

# Define label mappings (as per SoccerNet MVFoul task description)
SEVERITY_LABELS = {"No Offence": 0, "Offence + No Card": 1, "Offence + Yellow Card": 2, "Offence + Red Card": 3}
ACTION_TYPE_LABELS = {
    "Standing Tackle": 0, "Tackle": 1, "Holding": 2, "Pushing": 3,
    "Challenge": 4, "Dive": 5, "High Leg": 6, "Elbowing": 7
}
# Inverse maps for potential debugging or inspection
INV_SEVERITY_LABELS = {v: k for k, v in SEVERITY_LABELS.items()}
INV_ACTION_TYPE_LABELS = {v: k for k, v in ACTION_TYPE_LABELS.items()}

# Define keys for categorical features
CONTACT_FIELD = "Contact"
BODYPART_FIELD = "Bodypart"
UPPER_BODYPART_FIELD = "Upper body part"
LOWER_BODYPART_FIELD = "Lower body part"  # Added missing field
MULTIPLE_FOULS_FIELD = "Multiple fouls"
TRY_TO_PLAY_FIELD = "Try to play"
TOUCH_BALL_FIELD = "Touch ball"
HANDBALL_FIELD = "Handball"
HANDBALL_OFFENCE_FIELD = "Handball offence"
REPLAY_SPEED_FIELD = "Replay speed"
UNKNOWN_TOKEN = "<UNK>"

# Standard mappings for common fields (for reference and consistent values)
OFFENCE_VALUES = {"Offence": 1, "No offence": 0}
CONTACT_VALUES = {"With contact": 1, "Without contact": 0}
BODYPART_VALUES = {"Upper body": 1, "Lower body": 2}
UPPER_BODYPART_VALUES = {"Use of shoulder": 1, "Use of arm": 2}
LOWER_BODYPART_VALUES = {"Use of leg": 1, "Use of knee": 2, "Use of foot": 3}
MULTIPLE_FOULS_VALUES = {"Yes": 1, "No": 0}
TRY_TO_PLAY_VALUES = {"Yes": 1, "No": 0}
TOUCH_BALL_VALUES = {"Yes": 1, "No": 0}
HANDBALL_VALUES = {"Handball": 1, "No handball": 0}
HANDBALL_OFFENCE_VALUES = {"Yes": 1, "No": 0}

# Inverse mappings
INV_OFFENCE_VALUES = {v: k for k, v in OFFENCE_VALUES.items()}
INV_CONTACT_VALUES = {v: k for k, v in CONTACT_VALUES.items()}
INV_BODYPART_VALUES = {v: k for k, v in BODYPART_VALUES.items()}
INV_UPPER_BODYPART_VALUES = {v: k for k, v in UPPER_BODYPART_VALUES.items()}
INV_LOWER_BODYPART_VALUES = {v: k for k, v in LOWER_BODYPART_VALUES.items()}
INV_MULTIPLE_FOULS_VALUES = {v: k for k, v in MULTIPLE_FOULS_VALUES.items()}
INV_TRY_TO_PLAY_VALUES = {v: k for k, v in TRY_TO_PLAY_VALUES.items()}
INV_TOUCH_BALL_VALUES = {v: k for k, v in TOUCH_BALL_VALUES.items()}
INV_HANDBALL_VALUES = {v: k for k, v in HANDBALL_VALUES.items()}
INV_HANDBALL_OFFENCE_VALUES = {v: k for k, v in HANDBALL_OFFENCE_VALUES.items()}

class SoccerNetMVFoulDataset(Dataset):
    def __init__(self,
                 dataset_path: str, 
                 split: str, 
                 annotation_file_name: str = "annotations.json",
                 frames_per_clip: int = 24,
                 target_fps: int = 17,
                 load_all_views: bool = True,
                 max_views_to_load: int = None,
                 views_indices: list[int] = None,
                 transform=None,
                 target_height: int = 224,
                 target_width: int = 224):
        """
        Args:
            dataset_path (str): Path to the root of the SoccerNet MVFoul dataset (e.g., /path/to/SoccerNet_data/mvfouls).
                                This directory should contain split folders (train, valid, test).
            split (str): Dataset split, one of ['train', 'valid', 'test'].
            annotation_file_name (str): Name of the JSON file containing annotations, expected inside each split folder.
                                      E.g., <dataset_path>/<split>/<annotation_file_name>.
                                      Each entry should contain:
                                      - 'action_id': A unique identifier for the action.
                                      - 'video_files': A list of relative paths (from split_dir) to video files for different views.
                                      - 'labels': {'severity': 'No Offence', 'type': 'Tackle'}
                                      - 'start_frame' (optional): Start frame for the clip.
                                      - 'end_frame' (optional): End frame for the clip.
                                      - 'original_fps' (optional): FPS of the source video if resampling is needed.
            frames_per_clip (int): Number of frames to sample for each video clip.
            target_fps (int): Desired frames per second for the output clip.
            load_all_views (bool): If True (default), loads all available views for each action.
            max_views_to_load (int): Optional limit on number of views to load. If None (default), loads all available views.
            views_indices (list[int]): Specific indices of views to load from the 'video_files' list in annotations.
                                       If provided, overrides load_all_views and max_views_to_load.
            transform: PyTorch transforms to be applied to each clip.
            target_height (int): Target height for dummy tensors if video loading fails.
            target_width (int): Target width for dummy tensors if video loading fails.
        """
        self.dataset_path = Path(dataset_path)
        self.split = split
        self.split_dir = self.dataset_path / self.split
        self.annotation_path = self.split_dir / annotation_file_name

        self.frames_per_clip = frames_per_clip
        self.target_fps = target_fps

        self.load_all_views = load_all_views
        self.max_views_to_load = max_views_to_load
        self.views_indices = views_indices

        self.transform = transform
        self.target_height = target_height
        self.target_width = target_width

        if not self.annotation_path.exists():
            raise FileNotFoundError(f"Annotation file not found: {self.annotation_path}. "
                                    "Please ensure it exists or use SoccerNet's tools to generate/locate it.")
        
        raw_annotations_data = {}
        try:
            with open(self.annotation_path, 'r') as f:
                raw_annotations_data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Error decoding JSON from {self.annotation_path}: {e}")

        # Build vocabularies for all categorical features
        self.contact_vocab, self.num_contact_classes = self._build_vocab(raw_annotations_data, CONTACT_FIELD)
        self.bodypart_vocab, self.num_bodypart_classes = self._build_vocab(raw_annotations_data, BODYPART_FIELD)
        self.upper_bodypart_vocab, self.num_upper_bodypart_classes = self._build_vocab(raw_annotations_data, UPPER_BODYPART_FIELD)
        self.lower_bodypart_vocab, self.num_lower_bodypart_classes = self._build_vocab(raw_annotations_data, LOWER_BODYPART_FIELD)
        self.multiple_fouls_vocab, self.num_multiple_fouls_classes = self._build_vocab(raw_annotations_data, MULTIPLE_FOULS_FIELD)
        self.try_to_play_vocab, self.num_try_to_play_classes = self._build_vocab(raw_annotations_data, TRY_TO_PLAY_FIELD)
        self.touch_ball_vocab, self.num_touch_ball_classes = self._build_vocab(raw_annotations_data, TOUCH_BALL_FIELD)
        self.handball_vocab, self.num_handball_classes = self._build_vocab(raw_annotations_data, HANDBALL_FIELD)
        self.handball_offence_vocab, self.num_handball_offence_classes = self._build_vocab(raw_annotations_data, HANDBALL_OFFENCE_FIELD)
        
        print(f"Built '{CONTACT_FIELD}' vocab ({self.num_contact_classes} classes): {self.contact_vocab}")
        print(f"Built '{BODYPART_FIELD}' vocab ({self.num_bodypart_classes} classes): {self.bodypart_vocab}")
        print(f"Built '{UPPER_BODYPART_FIELD}' vocab ({self.num_upper_bodypart_classes} classes): {self.upper_bodypart_vocab}")
        print(f"Built '{LOWER_BODYPART_FIELD}' vocab ({self.num_lower_bodypart_classes} classes): {self.lower_bodypart_vocab}")
        print(f"Built '{MULTIPLE_FOULS_FIELD}' vocab ({self.num_multiple_fouls_classes} classes): {self.multiple_fouls_vocab}")
        print(f"Built '{TRY_TO_PLAY_FIELD}' vocab ({self.num_try_to_play_classes} classes): {self.try_to_play_vocab}")
        print(f"Built '{TOUCH_BALL_FIELD}' vocab ({self.num_touch_ball_classes} classes): {self.touch_ball_vocab}")
        print(f"Built '{HANDBALL_FIELD}' vocab ({self.num_handball_classes} classes): {self.handball_vocab}")
        print(f"Built '{HANDBALL_OFFENCE_FIELD}' vocab ({self.num_handball_offence_classes} classes): {self.handball_offence_vocab}")

        self.actions = self._process_annotations(raw_annotations_data)

        if not self.actions:
            print(f"Warning: No actions loaded from {self.annotation_path} after processing. Check the annotation file format and content.")

    def _build_vocab(self, all_actions_data: dict, field_name: str, unknown_token: str = UNKNOWN_TOKEN):
        unique_values = set()
        actions_dict = all_actions_data.get("Actions", {})
        if not isinstance(actions_dict, dict):
            print(f"Warning: 'Actions' key not found or not a dict in annotation data while building vocab for '{field_name}'.")
            # Fallback to just unknown token if no actions found
            vocab = {unknown_token: 0}
            return vocab, len(vocab)

        for action_details in actions_dict.values():
            if isinstance(action_details, dict):
                value = action_details.get(field_name)
                if value is not None and isinstance(value, str) and value.strip(): # Ensure it's a non-empty string
                    unique_values.add(value)
                
                # Also check in clips for replay speed
                if field_name == REPLAY_SPEED_FIELD and "Clips" in action_details:
                    for clip in action_details["Clips"]:
                        if isinstance(clip, dict) and REPLAY_SPEED_FIELD in clip:
                            replay_speed = str(clip.get(REPLAY_SPEED_FIELD))
                            if replay_speed and replay_speed.strip():
                                unique_values.add(replay_speed)
        
        sorted_values = sorted(list(unique_values))
        vocab = {unknown_token: 0} # UNK token gets index 0
        for i, value in enumerate(sorted_values):
            vocab[value] = i + 1 # Actual values start from 1
        return vocab, len(vocab)

    def _process_annotations(self, annotations_data: dict):
        processed_actions = []

        # Mappings from your specific JSON values to the internal label strings
        # For Severity:
        # "1.0" -> "Offence + No Card"
        # "3.0" -> "Offence + Yellow Card"
        # "5.0" -> "Offence + Red Card"
        # "" (empty string) or main "Offence" field is "No offence" -> "No Offence"
        severity_json_map = {
            "1.0": "Offence + No Card",
            "3.0": "Offence + Yellow Card",
            "5.0": "Offence + Red Card",
            "": "No Offence" 
        }

        # For Action Class:
        action_class_json_map = {
            "Challenge": "Challenge",
            "Tackling": "Tackle",
            "Standing tackling": "Standing Tackle",
            "High leg": "High Leg",
            "Dive": "Dive",
            "Pushing": "Pushing", # Added based on common foul types
            "Holding": "Holding",   # Added based on common foul types
            "Elbowing": "Elbowing" # Added based on common foul types
            # Add other mappings if present in your JSON, e.g. "Handball foul" -> ?
        }

        actions_dict = annotations_data.get("Actions")
        if not isinstance(actions_dict, dict):
            print(f"Warning: 'Actions' key not found in {self.annotation_path} or is not a dictionary. No actions will be loaded.")
            return []

        for action_id_str, action_details in actions_dict.items():
            if not isinstance(action_details, dict):
                # print(f"Warning: Action details for ID {action_id_str} is not a dictionary. Skipping.")
                continue

            # --- Severity Label ---
            json_severity_val = action_details.get("Severity", "") # Default to empty string if missing
            
            # If main "Offence" field is "No offence", it should always be "No Offence" severity
            if action_details.get("Offence") == "No offence":
                 severity_label_str = "No Offence"
            elif json_severity_val in severity_json_map:
                severity_label_str = severity_json_map[json_severity_val]
            else:
                # print(f"Warning: Unknown 'Severity' value '{json_severity_val}' for action {action_id_str}. Defaulting to 'No Offence'.")
                severity_label_str = "No Offence" # Default or skip

            if severity_label_str not in SEVERITY_LABELS:
                # This case should ideally be handled by the map or a default above
                # print(f"Warning: Mapped severity '{severity_label_str}' not in SEVERITY_LABELS for action {action_id_str}. Skipping.")
                continue
            
            numerical_severity = SEVERITY_LABELS[severity_label_str]

            # --- Action Type Label ---
            json_action_class = action_details.get("Action class")
            if json_action_class in action_class_json_map:
                action_type_label_str = action_class_json_map[json_action_class]
            else:
                # print(f"Warning: Unknown 'Action class' value '{json_action_class}' for action {action_id_str}. Skipping this action.")
                continue # Skip if action class is unknown and unmapped

            if action_type_label_str not in ACTION_TYPE_LABELS:
                # print(f"Warning: Mapped action type '{action_type_label_str}' not in ACTION_TYPE_LABELS for action {action_id_str}. Skipping.")
                continue
            
            numerical_action_type = ACTION_TYPE_LABELS[action_type_label_str]

            # --- Video Files ---
            clips_info_list = action_details.get("Clips", [])
            if not isinstance(clips_info_list, list) or not clips_info_list:
                # print(f"Warning: No 'Clips' found or 'Clips' is not a list for action {action_id_str}. Skipping.")
                continue
            
            video_files_relative = []
            # Path prefix to remove, specific to split. e.g. "Dataset/Train/"
            path_prefix_to_strip = f"Dataset/{self.split.capitalize()}/" 
            
            # Extract clip-specific information
            clip_replay_speeds = []
            
            for clip_info in clips_info_list:
                raw_url = clip_info.get("Url")
                if raw_url:
                    # Strip the "Dataset/Train/" or similar prefix
                    if raw_url.startswith(path_prefix_to_strip):
                        processed_url = raw_url[len(path_prefix_to_strip):]
                    else:
                        # print(f"Warning: Clip URL '{raw_url}' for action {action_id_str} does not start with expected prefix '{path_prefix_to_strip}'. Using as is, but might be wrong.")
                        processed_url = raw_url
                    
                    # Assume .mp4 extension if not present. Adjust if your files are .mkv or other.
                    if not Path(processed_url).suffix:
                        video_files_relative.append(processed_url + ".mp4")
                    else:
                        video_files_relative.append(processed_url)
                
                # Get replay speed for this clip
                replay_speed = clip_info.get(REPLAY_SPEED_FIELD, None)
                clip_replay_speeds.append(str(replay_speed) if replay_speed is not None else UNKNOWN_TOKEN)
            
            if not video_files_relative:
                # print(f"Warning: No valid video URLs extracted for action {action_id_str} after processing. Skipping.")
                continue

            # --- Original FPS (if available in annotation at the action level) ---
            annotated_original_fps_val = action_details.get("original_fps")
            final_original_fps = None 
            if annotated_original_fps_val is not None:
                try:
                    fps_val = float(annotated_original_fps_val)
                    if fps_val > 0:
                        final_original_fps = fps_val
                    else:
                        print(f"Warning: Invalid original_fps value \'{fps_val}\' (must be > 0) for action {action_id_str} at action level. Will be ignored.")
                except ValueError:
                    print(f"Warning: Non-float original_fps value \'{annotated_original_fps_val}\' for action {action_id_str} at action level. Will be ignored.")

            # --- Process all categorical features using standard mappings ---
            
            # Offence
            offence_str = action_details.get("Offence", UNKNOWN_TOKEN)
            offence_idx = OFFENCE_VALUES.get(offence_str, 0) # Default to 0 (No offence) if not found
            
            # Contact
            contact_str = action_details.get(CONTACT_FIELD, UNKNOWN_TOKEN)
            contact_idx = self.contact_vocab.get(contact_str, self.contact_vocab[UNKNOWN_TOKEN])
            # Standard mapping for consistent numerical values
            if contact_str in CONTACT_VALUES:
                contact_standard_idx = CONTACT_VALUES[contact_str]
            else:
                contact_standard_idx = 0 # Default to "Without contact"
            
            # Bodypart
            bodypart_str = action_details.get(BODYPART_FIELD, UNKNOWN_TOKEN)
            bodypart_idx = self.bodypart_vocab.get(bodypart_str, self.bodypart_vocab[UNKNOWN_TOKEN])
            # Standard mapping for consistent numerical values
            if bodypart_str in BODYPART_VALUES:
                bodypart_standard_idx = BODYPART_VALUES[bodypart_str]
            else:
                bodypart_standard_idx = 0 # Default to unknown
            
            # Upper body part (only applicable when Bodypart is "Upper body")
            upper_bodypart_str = action_details.get(UPPER_BODYPART_FIELD, UNKNOWN_TOKEN)
            upper_bodypart_idx = self.upper_bodypart_vocab.get(upper_bodypart_str, self.upper_bodypart_vocab[UNKNOWN_TOKEN])
            # Standard mapping
            if upper_bodypart_str in UPPER_BODYPART_VALUES:
                upper_bodypart_standard_idx = UPPER_BODYPART_VALUES[upper_bodypart_str]
            else:
                upper_bodypart_standard_idx = 0 # Default to unknown
                
            # Lower body part (only applicable when Bodypart is "Lower body")
            lower_bodypart_str = action_details.get(LOWER_BODYPART_FIELD, UNKNOWN_TOKEN)
            # Create vocab on the fly if this field wasn't explicitly processed
            if not hasattr(self, 'lower_bodypart_vocab'):
                self.lower_bodypart_vocab = {UNKNOWN_TOKEN: 0}
                self.num_lower_bodypart_classes = 1
                print(f"Warning: Creating lower_bodypart_vocab on the fly as it wasn't processed earlier.")
                
            lower_bodypart_idx = getattr(self, 'lower_bodypart_vocab', {}).get(lower_bodypart_str, 0)
            # Standard mapping
            if lower_bodypart_str in LOWER_BODYPART_VALUES:
                lower_bodypart_standard_idx = LOWER_BODYPART_VALUES[lower_bodypart_str]
            else:
                lower_bodypart_standard_idx = 0 # Default to unknown
            
            # Multiple fouls
            multiple_fouls_str = action_details.get(MULTIPLE_FOULS_FIELD, UNKNOWN_TOKEN)
            multiple_fouls_idx = self.multiple_fouls_vocab.get(multiple_fouls_str, self.multiple_fouls_vocab[UNKNOWN_TOKEN])
            # Standard mapping
            if multiple_fouls_str in MULTIPLE_FOULS_VALUES:
                multiple_fouls_standard_idx = MULTIPLE_FOULS_VALUES[multiple_fouls_str]
            else:
                multiple_fouls_standard_idx = 0 # Default to "No"
            
            # Try to play
            try_to_play_str = action_details.get(TRY_TO_PLAY_FIELD, UNKNOWN_TOKEN)
            try_to_play_idx = self.try_to_play_vocab.get(try_to_play_str, self.try_to_play_vocab[UNKNOWN_TOKEN])
            # Standard mapping
            if try_to_play_str in TRY_TO_PLAY_VALUES:
                try_to_play_standard_idx = TRY_TO_PLAY_VALUES[try_to_play_str]
            else:
                try_to_play_standard_idx = 0 # Default to "No"
            
            # Touch ball
            touch_ball_str = action_details.get(TOUCH_BALL_FIELD, UNKNOWN_TOKEN)
            touch_ball_idx = self.touch_ball_vocab.get(touch_ball_str, self.touch_ball_vocab[UNKNOWN_TOKEN])
            # Standard mapping
            if touch_ball_str in TOUCH_BALL_VALUES:
                touch_ball_standard_idx = TOUCH_BALL_VALUES[touch_ball_str]
            else:
                touch_ball_standard_idx = 0 # Default to "No"
            
            # Handball
            handball_str = action_details.get(HANDBALL_FIELD, UNKNOWN_TOKEN)
            handball_idx = self.handball_vocab.get(handball_str, self.handball_vocab[UNKNOWN_TOKEN])
            # Standard mapping
            if handball_str in HANDBALL_VALUES:
                handball_standard_idx = HANDBALL_VALUES[handball_str]
            else:
                handball_standard_idx = 0 # Default to "No handball"
            
            # Handball offence
            handball_offence_str = action_details.get(HANDBALL_OFFENCE_FIELD, UNKNOWN_TOKEN)
            handball_offence_idx = self.handball_offence_vocab.get(handball_offence_str, self.handball_offence_vocab[UNKNOWN_TOKEN])
            # Standard mapping
            if handball_offence_str in HANDBALL_OFFENCE_VALUES:
                handball_offence_standard_idx = HANDBALL_OFFENCE_VALUES[handball_offence_str]
            else:
                handball_offence_standard_idx = 0 # Default to "No"
            
            processed_actions.append({
                "action_id": action_id_str,
                "video_files_relative": video_files_relative,
                "label_severity": numerical_severity,
                "label_type": numerical_action_type,
                "original_fps_from_annotation": final_original_fps,
                
                # Original vocab indices (for compatibility with existing code)
                "contact_idx": contact_idx,
                "bodypart_idx": bodypart_idx,
                "upper_bodypart_idx": upper_bodypart_idx,
                "lower_bodypart_idx": lower_bodypart_idx,
                "multiple_fouls_idx": multiple_fouls_idx,
                "try_to_play_idx": try_to_play_idx,
                "touch_ball_idx": touch_ball_idx,
                "handball_idx": handball_idx,
                "handball_offence_idx": handball_offence_idx,
                
                # Standard indices (new)
                "offence_standard_idx": offence_idx,
                "contact_standard_idx": contact_standard_idx,
                "bodypart_standard_idx": bodypart_standard_idx,
                "upper_bodypart_standard_idx": upper_bodypart_standard_idx,
                "lower_bodypart_standard_idx": lower_bodypart_standard_idx,
                "multiple_fouls_standard_idx": multiple_fouls_standard_idx,
                "try_to_play_standard_idx": try_to_play_standard_idx,
                "touch_ball_standard_idx": touch_ball_standard_idx,
                "handball_standard_idx": handball_standard_idx,
                "handball_offence_standard_idx": handball_offence_standard_idx,
                
                # Additional data
                "clip_replay_speeds": clip_replay_speeds
            })
            
        if not processed_actions:
            print(f"Warning: No actions were successfully processed from {self.annotation_path}. Check mappings and file paths.")
            
        return processed_actions
        
    def __len__(self):
        return len(self.actions)

    def _get_video_clip(self, video_path_str: str, action_info: dict):
        video_path = self.split_dir / video_path_str
        if not video_path.exists():
            # print(f"Warning: Video file {video_path} not found for action {action_info['action_id']}. Skipping this view.")
            return None

        # --- Determine Original FPS (for informational purposes or if needed by transforms later) ---
        original_fps = action_info.get("original_fps_from_annotation") 
        if original_fps is None: 
            try:
                _, _, meta = read_video_timestamps(str(video_path), pts=[]) 
                original_fps = meta.get('video_fps')
                if original_fps is None or original_fps <= 0:
                    # print(f"Warning: Could not read valid FPS from metadata for {video_path} (got {original_fps}). Defaulting to None.")
                    original_fps = None # No reliable FPS found
            except Exception as e:
                # print(f"Warning: Could not read metadata for {video_path} due to {e}. Defaulting to None.")
                original_fps = None 
        
        # --- Load all frames from the video clip --- 
        try:
            # print(f"Attempting to read entire video {video_path}")
            vframes, aframes, info = read_video(str(video_path), pts_unit='sec', output_format="TCHW")
        except RuntimeError as e:
            # print(f"Error reading video {video_path}: {e}. Skipping this view for action {action_info['action_id']}.")
            return None

        if vframes.size(0) == 0:
            # print(f"Warning: No frames read from {video_path} for specified time range. Skipping view.")
            return None

        # Sample self.frames_per_clip frames
        num_loaded_frames = vframes.size(0)
        if num_loaded_frames < self.frames_per_clip:
            # print(f"Warning: Loaded only {num_loaded_frames} frames from {video_path}, which is less than frames_per_clip ({self.frames_per_clip}). Returning all loaded frames.")
            # Padding will be handled in __getitem__ if necessary
            sampled_frames = vframes 
        else:
            indices = torch.linspace(0, num_loaded_frames - 1, self.frames_per_clip).long()
            sampled_frames = torch.index_select(vframes, 0, indices) # (frames_per_clip, C, H, W)

        clip = sampled_frames.permute(1, 0, 2, 3) # (C, T, H, W) -> (C, frames_per_clip or num_loaded_frames, H, W)
        return clip

    def __getitem__(self, idx):
        action_info = self.actions[idx]
        
        all_action_video_files_relative = action_info["video_files_relative"]
        selected_video_paths_relative = []

        if self.views_indices is not None:
            # Use specific view indices if provided
            for i in self.views_indices:
                if 0 <= i < len(all_action_video_files_relative):
                    selected_video_paths_relative.append(all_action_video_files_relative[i])
        elif self.load_all_views:
            # Load all available views (default behavior)
            if self.max_views_to_load is not None:
                selected_video_paths_relative = all_action_video_files_relative[:self.max_views_to_load]
            else:
                selected_video_paths_relative = all_action_video_files_relative  # Use all available views
        else: 
            # Fallback: use max_views_to_load or default to 2
            limit = self.max_views_to_load if self.max_views_to_load is not None else 2
            selected_video_paths_relative = all_action_video_files_relative[:limit]
        
        if not selected_video_paths_relative: # Should not happen if _process_annotations filters properly
            # print(f"Warning: No views selected for action {action_info['action_id']}. Returning dummy.")
            num_expected_views = 1 # Fallback to 1
            if self.views_indices: num_expected_views = len(self.views_indices)
            elif self.load_all_views or self.max_views_to_load > 0 : num_expected_views = self.max_views_to_load
            
            dummy_clips = torch.zeros((num_expected_views, 3, self.frames_per_clip, self.target_height, self.target_width))
            
            # Return both original and standardized indices
            return {
                "clips": dummy_clips,
                "label_severity": torch.tensor(action_info["label_severity"], dtype=torch.long),
                "label_type": torch.tensor(action_info["label_type"], dtype=torch.long),
                
                # Original indices from vocabularies
                "contact_idx": torch.tensor(action_info["contact_idx"], dtype=torch.long),
                "bodypart_idx": torch.tensor(action_info["bodypart_idx"], dtype=torch.long),
                "upper_bodypart_idx": torch.tensor(action_info["upper_bodypart_idx"], dtype=torch.long),
                "lower_bodypart_idx": torch.tensor(action_info["lower_bodypart_idx"], dtype=torch.long),
                "multiple_fouls_idx": torch.tensor(action_info["multiple_fouls_idx"], dtype=torch.long),
                "try_to_play_idx": torch.tensor(action_info["try_to_play_idx"], dtype=torch.long),
                "touch_ball_idx": torch.tensor(action_info["touch_ball_idx"], dtype=torch.long),
                "handball_idx": torch.tensor(action_info["handball_idx"], dtype=torch.long),
                "handball_offence_idx": torch.tensor(action_info["handball_offence_idx"], dtype=torch.long),
                
                # Standardized indices based on fixed mappings
                "offence_standard_idx": torch.tensor(action_info["offence_standard_idx"], dtype=torch.long),
                "contact_standard_idx": torch.tensor(action_info["contact_standard_idx"], dtype=torch.long),
                "bodypart_standard_idx": torch.tensor(action_info["bodypart_standard_idx"], dtype=torch.long),
                "upper_bodypart_standard_idx": torch.tensor(action_info["upper_bodypart_standard_idx"], dtype=torch.long),
                "lower_bodypart_standard_idx": torch.tensor(action_info["lower_bodypart_standard_idx"], dtype=torch.long),
                "multiple_fouls_standard_idx": torch.tensor(action_info["multiple_fouls_standard_idx"], dtype=torch.long),
                "try_to_play_standard_idx": torch.tensor(action_info["try_to_play_standard_idx"], dtype=torch.long),
                "touch_ball_standard_idx": torch.tensor(action_info["touch_ball_standard_idx"], dtype=torch.long),
                "handball_standard_idx": torch.tensor(action_info["handball_standard_idx"], dtype=torch.long),
                "handball_offence_standard_idx": torch.tensor(action_info["handball_offence_standard_idx"], dtype=torch.long)
            }

        clips_for_action = []
        for video_path_rel_str in selected_video_paths_relative:
            clip = self._get_video_clip(video_path_rel_str, action_info)
            if clip is not None:
                if self.transform: # Transform expects (C, T, H, W)
                    clip = self.transform(clip)
                clips_for_action.append(clip)
        
        num_successfully_loaded = len(clips_for_action)
        num_expected_views = len(selected_video_paths_relative)

        final_clips_tensor = None
        if num_successfully_loaded == 0:
            print(f"Warning: All views failed to load for action {action_info['action_id']}. Returning dummy tensor.")
            # Create a dummy tensor of shape (num_expected_views, C, T, H, W)
            final_clips_tensor = torch.zeros((num_expected_views, 3, self.frames_per_clip, self.target_height, self.target_width))
        elif num_successfully_loaded < num_expected_views:
            print(f"Warning: Loaded {num_successfully_loaded}/{num_expected_views} views for action {action_info['action_id']}. Padding with dummies.")
            dummy_view_clip = torch.zeros((3, self.frames_per_clip, self.target_height, self.target_width)) # C, T, H, W
            if self.transform and clips_for_action: # try to match transformed shape
                 dummy_view_clip = torch.zeros_like(clips_for_action[0])

            for _ in range(num_expected_views - num_successfully_loaded):
                clips_for_action.append(dummy_view_clip)
            final_clips_tensor = torch.stack(clips_for_action)
        else:
            final_clips_tensor = torch.stack(clips_for_action)

        # Return both original and standardized indices in a dictionary
        return {
            "clips": final_clips_tensor,
            "label_severity": torch.tensor(action_info["label_severity"], dtype=torch.long),
            "label_type": torch.tensor(action_info["label_type"], dtype=torch.long),
            
            # Original indices from vocabularies
            "contact_idx": torch.tensor(action_info["contact_idx"], dtype=torch.long),
            "bodypart_idx": torch.tensor(action_info["bodypart_idx"], dtype=torch.long),
            "upper_bodypart_idx": torch.tensor(action_info["upper_bodypart_idx"], dtype=torch.long),
            "lower_bodypart_idx": torch.tensor(action_info["lower_bodypart_idx"], dtype=torch.long),
            "multiple_fouls_idx": torch.tensor(action_info["multiple_fouls_idx"], dtype=torch.long),
            "try_to_play_idx": torch.tensor(action_info["try_to_play_idx"], dtype=torch.long),
            "touch_ball_idx": torch.tensor(action_info["touch_ball_idx"], dtype=torch.long),
            "handball_idx": torch.tensor(action_info["handball_idx"], dtype=torch.long),
            "handball_offence_idx": torch.tensor(action_info["handball_offence_idx"], dtype=torch.long),
            
            # Standardized indices based on fixed mappings
            "offence_standard_idx": torch.tensor(action_info["offence_standard_idx"], dtype=torch.long),
            "contact_standard_idx": torch.tensor(action_info["contact_standard_idx"], dtype=torch.long),
            "bodypart_standard_idx": torch.tensor(action_info["bodypart_standard_idx"], dtype=torch.long),
            "upper_bodypart_standard_idx": torch.tensor(action_info["upper_bodypart_standard_idx"], dtype=torch.long),
            "lower_bodypart_standard_idx": torch.tensor(action_info["lower_bodypart_standard_idx"], dtype=torch.long),
            "multiple_fouls_standard_idx": torch.tensor(action_info["multiple_fouls_standard_idx"], dtype=torch.long),
            "try_to_play_standard_idx": torch.tensor(action_info["try_to_play_standard_idx"], dtype=torch.long),
            "touch_ball_standard_idx": torch.tensor(action_info["touch_ball_standard_idx"], dtype=torch.long),
            "handball_standard_idx": torch.tensor(action_info["handball_standard_idx"], dtype=torch.long),
            "handball_offence_standard_idx": torch.tensor(action_info["handball_offence_standard_idx"], dtype=torch.long)
        }

def variable_views_collate_fn(batch):
    """
    Custom collate function to handle batches with variable numbers of views.
    
    Args:
        batch: List of dictionaries from dataset __getitem__
        
    Returns:
        Dictionary with batched tensors, handling variable view counts
    """
    if not batch:
        return {}
    
    # Separate clips from other features
    clips_list = [item["clips"] for item in batch]
    
    # Get all other keys (excluding clips)
    other_keys = [key for key in batch[0].keys() if key != "clips"]
    
    # Stack other features normally (they should all have same batch dimension)
    batched_data = {}
    for key in other_keys:
        batched_data[key] = torch.stack([item[key] for item in batch])
    
    # Handle variable views for clips
    # Each item in clips_list has shape (num_views, C, T, H, W)
    # We need to create a list of tensors, where each tensor is (B, C, T, H, W) for one view
    
    max_views = max(clips.shape[0] for clips in clips_list)
    batch_size = len(clips_list)
    
    if max_views == 1 or all(clips.shape[0] == clips_list[0].shape[0] for clips in clips_list):
        # All items have same number of views, can stack normally
        batched_data["clips"] = torch.stack(clips_list)
    else:
        # Variable number of views - create list of view tensors
        # Pad shorter sequences with zeros
        C, T, H, W = clips_list[0].shape[1:]  # Get dimensions from first item
        
        # Create padded tensor: (batch_size, max_views, C, T, H, W)
        padded_clips = torch.zeros(batch_size, max_views, C, T, H, W, dtype=clips_list[0].dtype)
        
        for i, clips in enumerate(clips_list):
            num_views = clips.shape[0]
            padded_clips[i, :num_views] = clips
        
        batched_data["clips"] = padded_clips
        
        # Also create a mask indicating which views are real vs padded
        view_mask = torch.zeros(batch_size, max_views, dtype=torch.bool)
        for i, clips in enumerate(clips_list):
            num_views = clips.shape[0]
            view_mask[i, :num_views] = True
        
        batched_data["view_mask"] = view_mask
    
    return batched_data
