  # MAI 645 Final Project - Character Motion Prediction

  This repository contains the implementation for the MAI 645 final project on character motion prediction. The project compares three motion representations:

  1. Positional representation
  2. Euler-angle representation
  3. Quaternion representation

  The required preprocessing, training, and synthesis scripts are located in `code/`.

  ## Training Scripts

  ```bash
  python code/pytorch_train_pos_aclstm.py
  python code/pytorch_train_euler_aclstm.py
  python code/pytorch_train_quad_aclstm.py
  ```

  ## Synthesis Scripts

  ```bash
  python code/synthise_pos_motion.py
  python code/synthise_euler_motion.py
  python code/synthise_quad_motion.py
  ```

  ## Loss Functions

  The positional model uses MSE over all joint-position channels. The Euler-angle model uses MSE for the root translation and angle-distance loss for the joint rotations. The quaternion model uses MSE for the root translation and quaternion angular loss for the joint rotations.
