import tkinter as tk
from tkinter import filedialog, messagebox
import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from sklearn.cluster import DBSCAN

class App:
    def __init__(self, root):
        """
        Inicializa la aplicación GUI con botones y área de texto.

        Args:
            root (tk.Tk): La ventana principal de la aplicación.
        """
        self.root = root
        self.root.title("Análisis de Datos UMAP")
        
        # Botón para cargar archivo
        self.load_button = tk.Button(root, text="Cargar archivo .h5ad", command=self.load_file)
        self.load_button.pack(pady=10)
        
        # Área de texto para mostrar resultados
        self.text_area = tk.Text(root, wrap=tk.WORD, height=20, width=80)
        self.text_area.pack(pady=10)
        
        # Botón para mostrar gráfico
        self.plot_button = tk.Button(root, text="Mostrar gráfico UMAP", command=self.plot_data)
        self.plot_button.pack(pady=10)
        
        self.df = None
    
    def load_file(self):
        """
        Carga un archivo .h5ad, extrae las coordenadas UMAP y los identificadores de clústeres,
        y actualiza el DataFrame y el área de texto con la información.
        """
        file_path = filedialog.askopenfilename(filetypes=[("Archivos h5ad", "*.h5ad")])
        if not file_path:
            return
        
        try:
            # Cargar el archivo .h5ad
            adata = sc.read(file_path)
            
            # Extraer las coordenadas UMAP y el identificador del clúster
            umap_coords = adata.obsm['X_UMAP']
            cluster_ids = adata.obs.get('cluster_id', pd.Series([None] * umap_coords.shape[0]))
            
            # Crear un DataFrame para manejar los datos más fácilmente
            self.df = pd.DataFrame(umap_coords, columns=['UMAP1', 'UMAP2'])
            self.df['Cluster'] = cluster_ids
            
            # Verificar la presencia de datos en 'Cluster'
            self.text_area.delete(1.0, tk.END)
            self.text_area.insert(tk.END, "Primeras filas del DataFrame:\n")
            self.text_area.insert(tk.END, str(self.df.head()) + '\n')
            
            # Si 'Cluster' está vacío, asignar un valor predeterminado para prueba
            if self.df['Cluster'].isna().all():
                self.text_area.insert(tk.END, "\nTodos los valores en 'Cluster' son NaN. Asignando valores predeterminados.\n")
                self.df['Cluster'] = range(len(self.df))  # Asignar valores únicos para cada fila
            
            # Eliminar filas con valores NaN en 'Cluster'
            self.df = self.df.dropna(subset=['Cluster'])
            
            # Convertir 'Cluster' a enteros si es necesario
            self.df['Cluster'] = self.df['Cluster'].astype(int)
            
            # Imprimir el DataFrame final para verificar la conversión
            self.text_area.insert(tk.END, "\nDataFrame final con 'Cluster' como enteros:\n")
            self.text_area.insert(tk.END, str(self.df.head()) + '\n')
            
            # Imprimir coordenadas y clústeres como tuplas
            if not self.df.empty:
                self.text_area.insert(tk.END, "\nDatos de células:\n")
                for index, row in self.df.iterrows():
                    coord = (row['UMAP1'], row['UMAP2'])
                    cluster = row['Cluster']
                    self.text_area.insert(tk.END, f'Celula {index}: Coordenadas={coord}, Clúster={cluster}\n')
            else:
                self.text_area.insert(tk.END, "El DataFrame está vacío después de eliminar NaN.\n")
                
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo cargar el archivo: {e}")
    
    def plot_data(self):
        """
        Muestra un gráfico UMAP con clústeres usando DBSCAN y dibuja las envolturas convexas 
        para cada grupo de células.
        """
        if self.df is None or self.df.empty:
            messagebox.showerror("Error", "No se ha cargado ningún archivo o los datos están vacíos.")
            return
        
        # Realizar el clustering DBSCAN
        eps = 1.1  # Distancia máxima entre puntos para ser considerados en el mismo clúster
        min_samples = 5  # Número mínimo de puntos para formar un clúster
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(self.df[['UMAP1', 'UMAP2']])
        self.df['DBSCAN_Cluster'] = clustering.labels_
        
        fig, ax = plt.subplots(figsize=(8, 6))
        scatter = ax.scatter(self.df['UMAP1'], self.df['UMAP2'], c=self.df['DBSCAN_Cluster'], cmap='tab10', alpha=0.6)
        plt.xlabel('UMAP1')
        plt.ylabel('UMAP2')
        plt.title('Visualización de UMAP con Clústeres DBSCAN')
        plt.colorbar(scatter, label='Cluster ID')

        # Añadir envoltura convexa por grupo de clústeres
        self.plot_convex_hulls(ax)

        # Mostrar el gráfico en una ventana independiente
        plt.show()

    def plot_convex_hulls(self, ax):
        """
        Dibuja el Convex Hull para cada clúster en el gráfico.

        Args:
            ax (matplotlib.axes.Axes): El objeto de ejes en el que se dibujará el Convex Hull.
        """
        if not self.df.empty:
            # Agrupar datos por el clúster
            grouped = self.df.groupby('DBSCAN_Cluster')
            
            for cluster_id, group in grouped:
                points = group[['UMAP1', 'UMAP2']].values
                
                if len(points) >= 3:  # Se requiere al menos 3 puntos para formar un Convex Hull
                    hull = ConvexHull(points)
                    
                    # Dibujar los bordes del convex hull
                    for simplex in hull.simplices:
                        ax.plot(points[simplex, 0], points[simplex, 1], 'k-', alpha=0.7)

                    # Rellenar el área del convex hull
                    ax.fill(points[hull.vertices, 0], points[hull.vertices, 1], alpha=0.2, label=f'Cluster {cluster_id}')

            ax.legend()
            ax.grid(True)

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
