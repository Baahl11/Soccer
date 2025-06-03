"""
Módulo para procesar datos en lotes y mostrar el progreso.
"""
from typing import List, Callable, TypeVar, Any, Dict
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

T = TypeVar('T')

class BatchProcessor:
    """Clase para procesar datos en lotes con barra de progreso."""
    
    def __init__(self, batch_size: int = 10, max_workers: int = 5):
        """
        Inicializar el procesador por lotes.
        
        Args:
            batch_size: Tamaño del lote para procesar
            max_workers: Número máximo de hilos para procesar en paralelo
        """
        self.batch_size = batch_size
        self.max_workers = max_workers
    
    def process_batches(
        self,
        items: List[T],
        process_fn: Callable[[T], Any],
        desc: str = "Processing"
    ) -> List[Dict[str, Any]]:
        """
        Procesa una lista de elementos en lotes.
        
        Args:
            items: Lista de elementos a procesar
            process_fn: Función para procesar cada elemento
            desc: Descripción para la barra de progreso
            
        Returns:
            Lista de resultados procesados
        """
        results = []
        
        # Crear lotes
        batches = [
            items[i:i + self.batch_size]
            for i in range(0, len(items), self.batch_size)
        ]
        
        # Procesar lotes con barra de progreso
        with tqdm(total=len(items), desc=desc) as pbar:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                for batch in batches:
                    # Procesar el lote actual
                    batch_results = list(executor.map(process_fn, batch))
                    results.extend(batch_results)
                    
                    # Actualizar la barra de progreso
                    pbar.update(len(batch))
        
        return results
