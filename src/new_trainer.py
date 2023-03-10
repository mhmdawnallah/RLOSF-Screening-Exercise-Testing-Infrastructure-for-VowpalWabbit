from typing import List
from abc import ABC

class NewTrainer(ABC):
	def train(self, x: List[List[float]], y: List[float]):
		pass
	
	def predict(self, x: List[float]) -> float:
		pass

