# Monitor de Brecha de Género en STEM y Futuro del Trabajo

Proyecto WPower · ODS 5 (Igualdad de Género)

Este proyecto cuantifica el riesgo de automatización que enfrentan las mujeres en empleos STEM en México. A partir de los microdatos de la ENOE y tres modelos internacionales de automatización, construimos un índice que permite ver cómo la transición tecnológica podría amplificar la brecha de género existente en estos campos.

---

**Integrantes:**
- Nayeli Esmeralda Pantoja Flores
- Guillermo Díaz Esquivel
- Paola Nayeli Martinez Ibarra

---

## El Índice de Riesgo de Género (IRG)

El IRG nace de un problema concreto: las mujeres representan apenas el 12.9% de la ocupación STEM en México (CIEP, 2024), y la literatura internacional documenta un diferencial de riesgo de automatización de aproximadamente 2 puntos porcentuales a su cargo (Brussevich et al., FMI, 2018). Sin embargo, no existía una métrica que operacionalizara esto con datos laborales mexicanos a nivel de ocupación.

Lo que hacemos es cruzar los códigos SINCO de la ENOE con probabilidades de automatización de tres fuentes distintas y agregar un componente de género. La formulación quedó así:

```
IRG = P_Frey-Osborne × β_Anthropic × W_ola × (1 + δ_género) × Gallup_factor
```

Desglosando cada término:

| Componente | Qué captura | Fuente |
|---|---|---|
| `P_FO` | Probabilidad de computerización por ocupación | Frey & Osborne (2017), 702 ocupaciones |
| `β_obs` | Exposición observada (no teórica) a herramientas de IA | Massenkoff & McCrory / Anthropic (2026) |
| `W_ola` | Peso temporal según la ola de automatización (algorítmica, aumento, IA generativa) | McKinsey MGI / OCDE (2023-2024) |
| `δ_género` | Ajuste de +0.02 si el registro es mujer (+2pp) | Brussevich et al., FMI (2018) |
| `Gallup` | Factor de vulnerabilidad asociado a engagement laboral | State of the Global Workplace (2025) |

El producto de los tres primeros términos da el riesgo base de automatización por ocupación. El factor de género lo ajusta hacia arriba para mujeres, y el factor de Gallup introduce una dimensión de vulnerabilidad laboral que los modelos puramente tecnológicos suelen ignorar.

---

## Estructura del repositorio

```
wpower-stem-irg/
│
├── datos/
│   ├── datos_finales_stem.csv        # Microdatos ENOE procesados, estos son datos publicos  y no se agregan porque la interfaz visual de Github no lo permitió
│   └── tabla_irg_v3.csv              # IRG calculado por ocupación
│
├── notebooks/
│   └── 003_datos_ocup_sexo.ipynb     # Limpieza y procesamiento ENOE
│
├── irg_v3_enoe_real.py               # Script principal
├── irg_v3_enoe_real.png              # Dashboard de resultados, aun no está terminado
└── README.md
```

---

## Reproducción

### Dependencias

```bash
pip install pandas numpy matplotlib scipy
```

### Ejecución

El CSV `datos_finales_stem.csv` debe estar en el mismo directorio que el script principal. Luego:

```bash
python irg_v3_enoe_real.py
```

Esto genera tres salidas:
- `irg_v3_enoe_real.png` — dashboard con 6 gráficas de distribución y comparación
- `tabla_irg_v3.csv` — IRG desagregado por ocupación con sus fuentes
- `datos_finales_stem_con_irg.csv` — microdatos originales con las columnas IRG añadidas

### Reutilizar con otros datos ENOE

```python
from irg_v3_enoe_real import aplicar_irg

# Requiere columnas p3 (SINCO) y sex (1=hombre, 2=mujer)
resultado = aplicar_irg(tu_df, año=2026)
```

---

## Fuentes de datos

| Fuente | Uso en el proyecto | Año |
|---|---|---|
| ENOE / INEGI | Microdatos de ocupación y sexo (SINCO 13, 16, 22, 26) | 2020–2025 |
| CIEP | Participación femenina en STEM: 12.9% total, 15.5% en TIC | 2024 |
| CEPAL | Contexto de riesgo de automatización en México | 2024 |
| OCDE Employment Outlook | 27% de empleos OCDE en alto riesgo automatización | 2023 |
| OCDE JCLED | Diferencial de género en exposición a IA generativa | 2024 |
| UNESCO UIS | 35% graduadas STEM, <25% en empleos del campo | 2024 |
| Frey & Osborne | Probabilidades de computerización (702 ocupaciones) | 2017 |
| Anthropic (Massenkoff & McCrory) | Exposición observada a IA por ocupación | 2026 |
| FMI (Brussevich et al.) | Diferencial de género +2pp en riesgo automatización | 2018 |
| Gallup | Factor de vulnerabilidad por engagement laboral | 2025 |

---

## Nota sobre el ODS 5

Si la automatización elimina o transforma desproporcionadamente los empleos donde ya están las mujeres, la brecha se ensancha desde el lado de la retención laboral. Este índice pretende aportar evidencia para el debate.
