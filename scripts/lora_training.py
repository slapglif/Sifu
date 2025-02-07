from typing import List, Dict, Optional, Any, Literal, cast, Union, TYPE_CHECKING
from datetime import datetime
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.documents import Document
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerBase
)
from datasets import Dataset
from peft.tuners.lora import LoraConfig
from peft.mapping import get_peft_model
from peft.utils.other import prepare_model_for_kbit_training
from peft.utils.peft_types import TaskType
from peft.peft_model import PeftModel
import torch
from loguru import logger
from .logging_config import log_error_with_traceback

if TYPE_CHECKING:
    BasePreTrainedModel = PreTrainedModel
else:
    BasePreTrainedModel = object

class TrainingExample(BaseModel):
    """Schema for training examples"""
    input_text: str = Field(description="Input text for training")
    output_text: str = Field(description="Expected output text")
    metadata: Dict = Field(description="Additional metadata")
    quality_score: float = Field(description="Quality score of the example")

class TrainingMetrics(BaseModel):
    """Schema for training metrics"""
    loss: float = Field(description="Training loss")
    eval_loss: Optional[float] = Field(description="Evaluation loss")
    train_samples: int = Field(description="Number of training samples")
    eval_samples: Optional[int] = Field(description="Number of evaluation samples")
    training_time: float = Field(description="Training time in seconds")

class ModelAdaptationMetrics(BaseModel):
    """Schema for model adaptation metrics"""
    base_performance: Dict[str, float] = Field(description="Base model performance")
    adapted_performance: Dict[str, float] = Field(description="Adapted model performance")
    improvement: Dict[str, float] = Field(description="Performance improvement")

class LoRATrainingConfig(BaseModel):
    """Configuration for LoRA training"""
    model_name: str = Field(description="Base model name")
    lora_alpha: int = Field(description="LoRA alpha parameter")
    lora_dropout: float = Field(description="LoRA dropout rate")
    r: int = Field(description="LoRA rank")
    bias: Literal["none", "all", "lora_only"] = Field(description="LoRA bias type")
    task_type: str = Field(description="Task type for LoRA")
    target_modules: List[str] = Field(description="Target modules for LoRA")
    inference_mode: bool = Field(description="Whether to use inference mode")

class LoRATrainer:
    """Trainer class for LoRA fine-tuning"""
    def __init__(self, config: LoRATrainingConfig):
        self.config = config
        self.model: Optional[Union[BasePreTrainedModel, PeftModel]] = None
        self.tokenizer: Optional[PreTrainedTokenizerBase] = None
        self.trainer: Optional[Trainer] = None
        self._setup_model()
        
    def _setup_model(self) -> None:
        """Setup model and tokenizer"""
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
            if self.tokenizer is None:
                raise ValueError("Failed to load tokenizer")
            
            # Load model
            base_model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            if base_model is None:
                raise ValueError("Failed to load model")
                
            # Prepare model for k-bit training
            prepared_model = prepare_model_for_kbit_training(base_model)
            
            # Create LoRA config
            lora_config = LoraConfig(
                r=self.config.r,
                lora_alpha=self.config.lora_alpha,
                target_modules=self.config.target_modules,
                lora_dropout=self.config.lora_dropout,
                bias=self.config.bias,
                task_type=TaskType.CAUSAL_LM,
            )
            
            # Get PEFT model
            peft_model = get_peft_model(prepared_model, lora_config)
            if peft_model is None:
                raise ValueError("Failed to create PEFT model")
            self.model = cast(PeftModel, peft_model)
            
            logger.info(f"Model setup complete with config: {self.config}")
            
        except Exception as e:
            logger.error(f"Error setting up model: {e}")
            raise
            
    def prepare_training_data(self, examples: List[TrainingExample]) -> Dataset:
        """Prepare training data from examples"""
        try:
            if self.tokenizer is None:
                raise ValueError("Tokenizer not initialized")
                
            # Convert examples to dataset format
            dataset_dict = {
                "input_text": [],
                "output_text": [],
                "quality_score": []
            }
            
            # Ensure we have at least one example
            if not examples:
                raise ValueError("No training examples provided")
            
            for example in examples:
                dataset_dict["input_text"].append(example.input_text)
                dataset_dict["output_text"].append(example.output_text)
                dataset_dict["quality_score"].append(example.quality_score)
                
            # Create dataset
            dataset = Dataset.from_dict(dataset_dict)
            
            # Tokenize dataset
            def tokenize_function(examples):
                if self.tokenizer is None:
                    raise ValueError("Tokenizer not initialized")
                    
                # Combine input and output with appropriate format
                full_texts = [
                    f"Input: {input_text}\nOutput: {output_text}"
                    for input_text, output_text in zip(examples["input_text"], examples["output_text"])
                ]
                tokenized = self.tokenizer(
                    full_texts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors=None  # Return a dictionary instead of tensors
                )
                return {
                    "input_ids": tokenized["input_ids"],
                    "attention_mask": tokenized["attention_mask"]
                }
                
            tokenized_dataset = dataset.map(
                tokenize_function,
                batched=True,
                remove_columns=dataset.column_names
            )
            
            # Ensure we have at least one sample
            if len(tokenized_dataset) == 0:
                raise ValueError("No valid samples in dataset after tokenization")
            
            return tokenized_dataset
            
        except Exception as e:
            logger.error(f"Error preparing training data: {e}")
            raise
            
    def train(self, 
             train_dataset: Dataset,
             eval_dataset: Optional[Dataset] = None,
             num_train_epochs: int = 3,
             per_device_train_batch_size: int = 8,
             learning_rate: float = 2e-5,
             weight_decay: float = 0.01,
             max_grad_norm: float = 1.0) -> TrainingMetrics:
        """Train the model using LoRA"""
        try:
            if self.model is None:
                raise ValueError("Model not initialized")
            if self.tokenizer is None:
                raise ValueError("Tokenizer not initialized")
                
            # Setup training arguments
            training_args = TrainingArguments(
                output_dir=f"lora_adapters/{self.config.model_name}",
                num_train_epochs=num_train_epochs,
                per_device_train_batch_size=per_device_train_batch_size,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                max_grad_norm=max_grad_norm,
                logging_dir="logs",
                logging_steps=10,
                evaluation_strategy="epoch" if eval_dataset else "no",
                save_strategy="epoch",
                load_best_model_at_end=True if eval_dataset else False,
            )
            
            # Create data collator
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False
            )
            
            # Initialize trainer
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=data_collator,
            )
            
            # Train model
            train_result = trainer.train()
            
            # Return metrics
            return TrainingMetrics(
                loss=train_result.training_loss,
                eval_loss=train_result.metrics.get("eval_loss"),
                train_samples=len(train_dataset),
                eval_samples=len(eval_dataset) if eval_dataset else None,
                training_time=train_result.metrics.get("training_time", 0.0)
            )
            
        except Exception as e:
            logger.error(f"Error during training: {e}")
            raise
            
    def evaluate_adaptation(self, eval_dataset: Dataset) -> ModelAdaptationMetrics:
        """Evaluate the model adaptation"""
        try:
            if self.model is None:
                raise ValueError("Model not initialized")
            if self.tokenizer is None:
                raise ValueError("Tokenizer not initialized")
                
            # Create a new trainer for evaluation
            eval_trainer = Trainer(
                model=self.model,
                args=TrainingArguments(
                    output_dir="eval_tmp",
                    per_device_eval_batch_size=8,
                    remove_unused_columns=False
                ),
                eval_dataset=eval_dataset,
                data_collator=DataCollatorForLanguageModeling(
                    tokenizer=cast(PreTrainedTokenizerBase, self.tokenizer),
                    mlm=False
                )
            )
            
            # Evaluate base model
            base_metrics = self._evaluate_model(self.model, eval_dataset)
            
            # Evaluate adapted model (using same model since we don't have a separate adapted model yet)
            adapted_metrics = base_metrics
            
            # Calculate improvements (will be 0 since we're using same model)
            improvements = {
                metric: 0.0
                for metric in base_metrics.keys()
            }
            
            return ModelAdaptationMetrics(
                base_performance=base_metrics,
                adapted_performance=adapted_metrics,
                improvement=improvements
            )
            
        except Exception as e:
            logger.error(f"Error evaluating adaptation: {e}")
            raise
            
    def _evaluate_model(self, model: Any, dataset: Dataset) -> Dict[str, float]:
        """Evaluate a model on a dataset"""
        try:
            if self.tokenizer is None:
                raise ValueError("Tokenizer not initialized")
                
            trainer = Trainer(
                model=model,
                eval_dataset=dataset,
                data_collator=DataCollatorForLanguageModeling(
                    tokenizer=self.tokenizer,
                    mlm=False
                )
            )
            
            metrics = trainer.evaluate()
            if metrics is None:
                raise ValueError("Model evaluation failed")
                
            return {
                "perplexity": torch.exp(torch.tensor(metrics["eval_loss"])).item(),
                "loss": metrics["eval_loss"]
            }
            
        except Exception as e:
            log_error_with_traceback(e, "Error in model evaluation")
            raise
            
    def save_adapter(self, path: str, adapter_name: str = "default"):
        """Save the LoRA adapter weights"""
        try:
            if self.model is None:
                raise ValueError("Model not initialized")
                
            self.model.save_pretrained(path)
            logger.info(f"Saved LoRA adapter to {path}")
        except Exception as e:
            logger.error(f"Error saving adapter: {e}")
            raise
            
    def load_adapter(self, path: str, adapter_name: str = "default"):
        """Load a saved LoRA adapter"""
        try:
            if self.model is None:
                raise ValueError("Model not initialized")
                
            self.model.load_adapter(path, adapter_name=adapter_name)
            logger.info(f"Loaded LoRA adapter from {path}")
        except Exception as e:
            logger.error(f"Error loading adapter: {e}")
            raise 