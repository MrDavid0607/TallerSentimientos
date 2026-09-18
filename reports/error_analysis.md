# Error Analysis

Se seleccionaron aleatoriamente 20 errores del modelo final utilizando semilla 42, incluyendo errores de ambas clases.

## Frecuencia por categoría

- **other**: 6
- **informal**: 4
- **intensification**: 3
- **sarcasm**: 2
- **negation**: 2
- **mixed**: 1
- **contrast**: 1
- **elongation**: 1

## Patrones principales

### 1. other

Esta categoría fue una de las más frecuentes en la muestra de errores. Los ejemplos muestran que el modelo puede equivocarse cuando la polaridad no está expresada únicamente mediante palabras claramente positivas o negativas, sino que depende del contexto completo de la oración.

### 2. informal

Este segundo patrón también aparece de forma recurrente. En estos casos, el lenguaje utilizado introduce información que una representación TF-IDF puede tener dificultades para interpretar correctamente, especialmente cuando el sentimiento depende de expresiones no literales, formas informales o contexto.

## Conclusión

El modelo final presenta un buen desempeño global, pero los errores analizados muestran limitaciones frente a fenómenos lingüísticos que requieren mayor comprensión contextual que la disponible mediante una representación TF-IDF clásica.
