# Atlas

<!-- generated comment that must not reach the index -->
Atlas cleans and joins messy spreadsheets into one tidy table. On the bundled sample of 40 sheets it repaired 31 and flagged 9 for review.

![diagram](docs/diagram.png)

## How it works
```mermaid
flowchart LR
  A --> B
```
It reads each sheet, normalises headers, and joins on a key column.

| Step | Time |
|---|---|
| read | 2 s |
