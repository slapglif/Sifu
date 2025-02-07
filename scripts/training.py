from typing import List, Dict, Optional

from loguru import logger

class Training:
    def train(self, examples: List[Dict], num_samples: Optional[int] = None) -> None:
        """Train the model on examples"""
        if not examples:
            logger.warning("No training examples provided")
            return
            
        # Ensure num_samples is valid
        if num_samples is None:
            num_samples = len(examples)
        else:
            num_samples = max(1, min(num_samples, len(examples)))  # Ensure between 1 and len(examples)
            
        logger.info(f"Training on {num_samples} examples")
        
        # Convert examples to training format
        training_data = []
        for example in examples[:num_samples]:
            try:
                formatted = self._format_example(example)
                if formatted:
                    training_data.append(formatted)
            except Exception as e:
                logger.warning(f"Failed to format example: {e}")
                
        if not training_data:
            logger.warning("No valid training examples after formatting")
            return
            
        try:
            # Train model
            self.model.train(training_data)
            logger.info("Training complete")
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise 