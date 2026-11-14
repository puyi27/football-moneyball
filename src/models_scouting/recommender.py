import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

class ScoutingRecommender:
    def __init__(self, df_metrics: pd.DataFrame):
        """
        Inicializa el recomendador filtrando únicamente a los jugadores fiables
        (aquellos que superan el umbral de minutos establecido en el pipeline de métricas).
        """
        self.df = df_metrics[df_metrics['is_reliable'] == True].copy()
        self.scaler = StandardScaler()
        
    def calculate_weighted_score(self, weights: dict) -> pd.DataFrame:
        """
        Genera un ranking basado en pesos personalizados por el scout.
        Ejemplo de weights: {'goals_p90': 1.0, 'assists_p90': 0.5}
        """
        self.df['custom_score'] = 0.0
        for metric, weight in weights.items():
            if metric in self.df.columns:
                self.df['custom_score'] += self.df[metric] * weight
                
        # Retorna el DataFrame ordenado de mayor a menor puntuación
        return self.df.sort_values(by='custom_score', ascending=False)
        
    def find_similar_players(self, player_id: int, features: list, k: int = 5) -> pd.DataFrame:
        """
        Encuentra los k jugadores más similares utilizando K-Nearest Neighbors 
        basado en la distancia euclidiana de las características seleccionadas.
        """
        if player_id not in self.df['player_id'].values:
            raise ValueError(f"El jugador ID {player_id} no existe o no tiene minutos suficientes para ser fiable.")
            
        # Extraer y escalar las características (fundamental para KNN)
        X = self.df[features]
        X_scaled = self.scaler.fit_transform(X)
        
        # Ajustar el modelo KNN
        knn = NearestNeighbors(n_neighbors=k+1, metric='euclidean')
        knn.fit(X_scaled)
        
        # Encontrar el índice del jugador objetivo
        target_idx = self.df.index[self.df['player_id'] == player_id].tolist()[0]
        target_scaled = X_scaled[target_idx].reshape(1, -1)
        
        # Calcular distancias e índices
        distances, indices = knn.kneighbors(target_scaled)
        
        # Excluir al propio jugador objetivo (siempre será la distancia 0, índice 0)
        similar_indices = indices[0][1:]
        similar_distances = distances[0][1:]
        
        # Construir el resultado
        df_similar = self.df.iloc[similar_indices].copy()
        df_similar['similarity_distance'] = similar_distances
        
        return df_similar.sort_values(by='similarity_distance', ascending=True)