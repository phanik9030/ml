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

function addLimitToCTEs(sql, limit = 100) {
  const limitText = `LIMIT ${limit}`;

  // Utility to recursively inject LIMIT into nested SELECTs
  function injectLimit(sqlBlock) {
    const selectRegex = /\bSELECT\b[\s\S]*?(?=(\bSELECT\b|\bFROM\b|$))/gi;
    let result = sqlBlock;
    let match;
    let insertPositions = [];

    // Find all SELECTs that are not already followed by a LIMIT in the same scope
    const selectBlocks = [];
    const stack = [];
    let current = '';
    let inQuote = false;
    let quoteChar = '';
    let i = 0;

    while (i < result.length) {
      const char = result[i];
      current += char;

      if ((char === `'` || char === `"`) && result[i - 1] !== '\\') {
        if (!inQuote) {
          inQuote = true;
          quoteChar = char;
        } else if (quoteChar === char) {
          inQuote = false;
        }
      }

      if (!inQuote) {
        if (char === '(') {
          stack.push(current.length - 1);
        } else if (char === ')') {
          const start = stack.pop();
          if (start !== undefined) {
            const block = current.slice(start, current.length);
            const inner = current.slice(start + 1, current.length - 1);
            if (
              /\bSELECT\b/i.test(inner) &&
              !/\bLIMIT\b\s+\d+/i.test(inner)
            ) {
              // inject limit inside this block
              const modifiedInner = injectLimit(inner.trim());
              const withLimit = `${modifiedInner} ${limitText}`;
              current = current.slice(0, start + 1) + withLimit + ')';
            }
          }
        }
      }

      i++;
    }

    // After handling nested, add LIMIT to top-level SELECT if not already there
    if (
      /\bSELECT\b/i.test(current) &&
      /\bFROM\b/i.test(current) &&
      !/\bLIMIT\b\s+\d+/i.test(current)
    ) {
      current = current.replace(/;?\s*$/, ` ${limitText};`);
    }

    return current;
  }

  return injectLimit(sql);
}
