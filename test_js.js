  function addLimitToCTEs(sql, limit = 100) {
    const limitText = `LIMIT ${limit}`;

    // Basic logic to detect SELECTs inside WITH or standalone
    const blocks = [];
    let i = 0;
    let depth = 0;
    let start = 0;
    const tokens = sql.split("");

    while (i < tokens.length) {
      if (tokens[i] === "(") {
        if (depth === 0) start = i;
        depth++;
      } else if (tokens[i] === ")") {
        depth--;
        if (depth === 0) {
          blocks.push({ start, end: i });
        }
      }
      i++;
    }

    // Insert LIMIT inside each detected SQL block if missing
    let result = sql;
    let offset = 0;

    blocks.forEach(({ start, end }) => {
      const block = result.slice(start + offset, end + 1 + offset);
      const hasLimit = /\blimit\s+\d+/i.test(block);

      if (!hasLimit && /\bselect\b/i.test(block)) {
        // Inject LIMIT before the closing parenthesis
        const modifiedBlock =
          block.replace(/\)\s*$/, ` ${limitText})`) || `${block} ${limitText}`;
        result =
          result.slice(0, start + offset) +
          modifiedBlock +
          result.slice(end + 1 + offset);
        offset += limitText.length + 1;
      }
    });

    // Also apply LIMIT to outer SELECT if needed
    const outerHasLimit = /\blimit\s+\d+\s*;?\s*$/i.test(result);
    if (!outerHasLimit && /\bselect\b/i.test(result)) {
      result = result.trim().replace(/;?$/, ` ${limitText};`);
    }

    return result;
  }
