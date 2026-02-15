# Potenzielle Probleme / Bugs

Diese Punkte wurden beim Aufräumen erkannt, aber **bewusst nicht funktional geändert**:

1. **Mögliche API-Inkompatibilität bei `resp.output_text`:**
   Der Code verlässt sich auf `resp.output_text` im Responses-Objekt. Je nach OpenAI-Python-Version/Response-Form kann das Feld fehlen oder anders strukturiert sein.

2. **Strikte Fehlerprüfung auf `resp.status == "completed"`:**
   Andere valide Zwischen-/Endzustände könnten auftreten (je nach API-Verhalten). Aktuell führt alles außer `completed` zu einem Fehler.

3. **`time.sleep(args.sleep)` auch bei `0`/sehr klein:**
   Bei großen Datensätzen kann ein unnötiger Delay die Laufzeit stark erhöhen, wenn der Default nicht angepasst wird.

4. **Keine Retries/Backoff bei transienten API-Fehlern:**
   Netz- oder Rate-Limit-Probleme führen direkt zu einem Fehler pro Frage.

5. **`key_map[final_topic_key]` wirft `KeyError` bei unerwartetem Modelloutput:**
   Auch wenn JSON-Schema den Enum einschränkt, kann bei Schema-/Provider-Anomalien ein Laufzeitfehler entstehen.

6. **`OPENAI_API_KEY` wird nur auf Existenz geprüft:**
   Leerer/ungültiger Key wird erst spät beim API-Call sichtbar.
