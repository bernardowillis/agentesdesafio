using System.Collections;
using System.Collections.Generic;
using System.IO;
using UnityEngine;

[System.Serializable]
public class QEntry
{
    public int x;
    public int y;
    public float[] qValues; // arriba, abajo, izquierda, derecha
}

[System.Serializable]
public class QData
{
    public QEntry[] rows;
}
// ------------------------------------

public class Qlearning : MonoBehaviour
{
    [Header("Configuración del Agente")]
    public string fileName = "q_table_unity.json";
    public float moveSpeed = 0.3f; 
    public Vector2Int goalPos = new Vector2Int(10, 10); // meta

    [Header("Configuración del Entorno")]
    public int gridWidth = 11;
    public int gridHeight = 11;
    public GameObject floorPrefab;
    public GameObject obstaclePrefab;

    // diccionario
    private Dictionary<string, float[]> brain = new Dictionary<string, float[]>();
    
    // set para búsqueda rápida de obstáculos
    private HashSet<Vector2Int> obstacles = new HashSet<Vector2Int>();

    void Start()
    {
        // definir obstáculos (Copiados de Python)
        DefineObstacles();

        // generar el entorno visualmente
        GenerateEnvironment();

        // cargar el cerebro y empezar a moverse
        LoadBrain();
        StartCoroutine(MoveRoutine());
    }

    // carga de datos

    void DefineObstacles()
    {
        // los mismos que el scipt de python
        obstacles.Add(new Vector2Int(0, 1)); obstacles.Add(new Vector2Int(1, 1));
        obstacles.Add(new Vector2Int(2, 1)); obstacles.Add(new Vector2Int(3, 1));
        obstacles.Add(new Vector2Int(4, 1)); obstacles.Add(new Vector2Int(6, 6));
        obstacles.Add(new Vector2Int(7, 6)); obstacles.Add(new Vector2Int(8, 6));
        obstacles.Add(new Vector2Int(9, 6)); obstacles.Add(new Vector2Int(6, 7));
        obstacles.Add(new Vector2Int(6, 8)); obstacles.Add(new Vector2Int(6, 10));
    }

    void GenerateEnvironment()
    {
        // contenedor
        GameObject gridParent = new GameObject("EnvironmentGrid");

        for (int x = 0; x < gridWidth; x++)
        {
            for (int y = 0; y < gridHeight; y++)
            {
                Vector2Int pos = new Vector2Int(x, y);
                // posición en el mundo
                Vector3 worldPos = new Vector3(x, y, 1); 

                GameObject tileToSpawn = obstacles.Contains(pos) ? obstaclePrefab : floorPrefab;
                
                GameObject spawnedTile = Instantiate(tileToSpawn, worldPos, Quaternion.identity);
                spawnedTile.transform.SetParent(gridParent.transform);

                // la meta se pinta en verde
                if (pos == goalPos && spawnedTile.GetComponent<Renderer>())
                    spawnedTile.GetComponent<Renderer>().material.color = Color.green;
            }
        }
        Debug.Log("Grid generado.");
    }

    void LoadBrain()
    {
        string path = Path.Combine(Application.streamingAssetsPath, fileName);
        if (File.Exists(path))
        {
            string jsonContent = File.ReadAllText(path);
            QData data = JsonUtility.FromJson<QData>(jsonContent);
            brain.Clear();
            foreach (QEntry entry in data.rows)
            {
                brain[entry.x + "," + entry.y] = entry.qValues;
            }
            Debug.Log($"Cargando estados: {brain.Count} estados.");
        }
        else
        {
            Debug.LogError("no se encontró el archivo JSON");
        }
    }

    // movimiento

    IEnumerator MoveRoutine()
    {
        yield return new WaitForSeconds(1f);

        int maxSteps = 200;
        int steps = 0;

        while (steps < maxSteps)
        {
            // determinar posición actual entera
            int currentX = Mathf.RoundToInt(transform.position.x);
            int currentY = Mathf.RoundToInt(transform.position.y);
            Vector2Int currentPos = new Vector2Int(currentX, currentY);

            // se verifica si ya llegamos a la meta
            if (currentPos == goalPos)
            {
                Debug.Log("¡LLEGÓ A LA META!");
                yield break;
            }

            string key = currentX + "," + currentY;

            if (brain.ContainsKey(key))
            {
                float[] values = brain[key];
                int action = GetBestAction(values);
                MoveAgent(action);
            }
            else
            {
                // si el agente cae en un estado que no entrenó
                Debug.LogWarning($"Estado desconocido ({key}) en paso {steps}. Agente detenido.");
                yield break;
            }

            steps++;
            yield return new WaitForSeconds(moveSpeed);
        }
        Debug.Log("Se alcanzaron los pasos máximos sin llegar a la meta.");
    }

    // se mapea, arriba abajo izquierda derecha
    void MoveAgent(int action)
    {
        Vector3 direction = Vector3.zero;
        switch (action)
        {
            case 0: direction = Vector3.up; break;
            case 1: direction = Vector3.down; break;
            case 2: direction = Vector3.left; break;
            case 3: direction = Vector3.right; break;
        }
        // se mueve 1 unidad exacta
        transform.position += direction;
    }

    int GetBestAction(float[] qValues)
    {
        float maxVal = float.MinValue;
        int bestIndex = 0;
        // encuentra el índice del valor más alto (Argmax)
        for (int i = 0; i < qValues.Length; i++)
        {
            if (qValues[i] > maxVal)
            {
                maxVal = qValues[i];
                bestIndex = i;
            }
        }
        return bestIndex;
    }
}