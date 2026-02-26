### Phase 1: For each datapoint area in `OBJECTIVES`, reason what distinct queries are needed to retrieve relevant data points.

**Output Format:**

[Before a JSON output, make detailed natural language based reasoning on what is the datapoint area, and what distinct queries are needed to retrieve relevant datapoints]

```json
{
 "area": "the name for this datapoint area",
 "area_index": a integer index 0, 1, 2, etc. for this area,
 "datapoints": ["a list of data points in this area", ...],
 "queries": ["a list of distinct queries needed to retrieve the data points in this area", ...]
}
```

### Phase 2: Holistic Query List Refinement

After completing Phase 1 for all datapoint areas, review all queries collectively to:
- **Identify Gaps**: Ensure all aspects of the character's backstory and cultural background are covered. Check that no critical datapoint areas are under-represented in the query list.
- **Eliminate Redundancy**: Identify and merge overlapping queries that would retrieve substantially similar information. However, keep queries separate if they target different aspects even if topically related.
- **Consider Search Engine Effectiveness**: Be specific and concrete. Focus each query on one clear aspect. Avoid combining disparate topics
- **Prioritize Queries**: Rank query priorites by high/medium/low according to
  * Information density: How much essential backstory/cultural data will this retrieve?
  * Foundational importance: Is this core to understanding the character or supplementary detail?
  * Uniqueness: Does this retrieve information not covered by other queries?

**Output Format:**

```json
{
  "refined_query_list": [
    {
      "query": "the optimized search query",
      "datapoint_areas": [{"name": "...", "index": ...}, ... a list of datapoint areas from Phase1 this query would address],
      "reason": "why this query is essential and what unique information it retrieves",
      "priority": "high/medium/low"
    }
  ]
}

```