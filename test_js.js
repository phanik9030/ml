  const { Parser } = require("node-sql-parser");
  const parser = new Parser();
  
  let ast = parser.astify(sql);

  // Function to recursively apply LIMIT 100 to all SELECT statements
  function applyLimitToSelect(astNode, limitValue = 100) {
    if (!astNode) return;

    if (Array.isArray(astNode)) {
      astNode.forEach((node) => applyLimitToSelect(node, limitValue));
    } else {
      if (astNode.type === "select" && !astNode.limit) {
        astNode.limit = {
          seperator: "",
          value: [{ type: "number", value: limitValue }],
        };
      }

      // Recursively handle with clauses (CTEs)
      if (astNode.with && astNode.with.clauses) {
        astNode.with.clauses.forEach((clause) =>
          applyLimitToSelect(clause.stmt, limitValue)
        );
      }

      // Handle subqueries
      if (astNode.from) {
        astNode.from.forEach((item) => {
          if (item.type === "subquery" && item.subquery) {
            applyLimitToSelect(item.subquery, limitValue);
          }
        });
      }
    }
  }

  // Apply limit to all select statements
  applyLimitToSelect(ast, 100);

  // Convert back to SQL
  const modifiedSql = parser.sqlify(ast);
