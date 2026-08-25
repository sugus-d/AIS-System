import { useState, useMemo } from "react";

export interface Column<T> {
  key: keyof T;
  label: string;
  width?: string;
  sortable?: boolean;
  render?: (value: any, row: T, index: number) => React.ReactNode;
  align?: "left" | "center" | "right";
}

interface DataTableProps<T> {
  columns: Column<T>[];
  data: T[];
  rowKey: keyof T;
  pageSize?: number;
  selectable?: boolean;
  onSelectionChange?: (selectedKeys: any[]) => void;
  onRowClick?: (row: T) => void;
  loading?: boolean;
  emptyMessage?: string;
}

export default function DataTable<T extends Record<string, any>>({
  columns,
  data,
  rowKey,
  pageSize = 10,
  selectable = false,
  onSelectionChange,
  onRowClick,
  loading = false,
  emptyMessage = "暂无数据",
}: DataTableProps<T>) {
  const [currentPage, setCurrentPage] = useState(1);
  const [sortKey, setSortKey] = useState<keyof T | null>(null);
  const [sortOrder, setSortOrder] = useState<"asc" | "desc">("asc");
  const [selectedRows, setSelectedRows] = useState<Set<any>>(new Set());

  // Handle sorting
  const handleSort = (key: keyof T) => {
    if (sortKey === key) {
      setSortOrder(sortOrder === "asc" ? "desc" : "asc");
    } else {
      setSortKey(key);
      setSortOrder("asc");
    }
  };

  // Sort and paginate data
  const processedData = useMemo(() => {
    let result = [...data];

    // Sort
    if (sortKey) {
      result.sort((a, b) => {
        const aVal = a[sortKey];
        const bVal = b[sortKey];

        if (aVal === null || aVal === undefined) return 1;
        if (bVal === null || bVal === undefined) return -1;

        if (typeof aVal === "string") {
          return sortOrder === "asc"
            ? aVal.localeCompare(bVal)
            : bVal.localeCompare(aVal);
        }

        if (typeof aVal === "number") {
          return sortOrder === "asc" ? aVal - bVal : bVal - aVal;
        }

        return 0;
      });
    }

    return result;
  }, [data, sortKey, sortOrder]);

  // Paginate
  const totalPages = Math.ceil(processedData.length / pageSize);
  const paginatedData = useMemo(() => {
    const start = (currentPage - 1) * pageSize;
    return processedData.slice(start, start + pageSize);
  }, [processedData, currentPage, pageSize]);

  // Handle selection
  const handleSelectRow = (key: any) => {
    const newSelected = new Set(selectedRows);
    if (newSelected.has(key)) {
      newSelected.delete(key);
    } else {
      newSelected.add(key);
    }
    setSelectedRows(newSelected);
    onSelectionChange?.([...newSelected]);
  };

  const handleSelectAll = () => {
    const newSelected = new Set<any>();
    if (selectedRows.size !== paginatedData.length) {
      paginatedData.forEach((row) => {
        newSelected.add(row[rowKey]);
      });
    }
    setSelectedRows(newSelected);
    onSelectionChange?.([...newSelected]);
  };

  if (loading) {
    return (
      <div className="card-base p-12 text-center">
        <div className="text-4xl mb-4">⏳</div>
        <p className="text-body text-[color:var(--color-text-secondary)]">
          加载中...
        </p>
      </div>
    );
  }

  if (processedData.length === 0) {
    return (
      <div className="card-base p-12 text-center">
        <div className="text-6xl mb-4">📭</div>
        <p className="text-body text-[color:var(--color-text-secondary)]">
          {emptyMessage}
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {/* Table */}
      <div className="card-base overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="bg-[color:var(--color-neutral)] border-b border-[color:var(--color-border)]">
                {selectable && (
                  <th className="w-12 px-4 py-3">
                    <input
                      type="checkbox"
                      className="w-4 h-4 rounded"
                      checked={
                        paginatedData.length > 0 &&
                        selectedRows.size === paginatedData.length
                      }
                      onChange={handleSelectAll}
                    />
                  </th>
                )}
                {columns.map((col) => (
                  <th
                    key={String(col.key)}
                    style={col.width ? { width: col.width } : undefined}
                    className={`px-6 py-3 text-body font-semibold text-[color:var(--color-text-primary)] ${col.align === "center"
                        ? "text-center"
                        : col.align === "right"
                          ? "text-right"
                          : "text-left"
                      } ${col.sortable ? "cursor-pointer hover:bg-[#f0f0f0]" : ""
                      } ${col.width ? `w-[${col.width}]` : ""}`}
                    onClick={() => col.sortable && handleSort(col.key)}
                  >
                    <div className={`flex items-center gap-2 ${col.align === "center"
                        ? "justify-center"
                        : col.align === "right"
                          ? "justify-end"
                          : ""
                      }`}>
                      {col.label}
                      {col.sortable && sortKey === col.key && (
                        <span className="text-xs">
                          {sortOrder === "asc" ? "▲" : "▼"}
                        </span>
                      )}
                    </div>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {paginatedData.map((row, idx) => {
                const rowKeyValue = row[rowKey];
                const isSelected = selectedRows.has(rowKeyValue);

                return (
                  <tr
                    key={String(rowKeyValue)}
                    className={`border-b border-[color:var(--color-border)] hover:bg-[color:var(--color-neutral)] transition-colors ${isSelected ? "bg-[color:var(--color-primary-light)]" : ""
                      } ${onRowClick ? "cursor-pointer" : ""}`}
                    onClick={() => onRowClick && onRowClick(row)}
                  >
                    {selectable && (
                      <td className="px-4 py-4">
                        <input
                          type="checkbox"
                          className="w-4 h-4 rounded"
                          checked={isSelected}
                          onChange={() => handleSelectRow(rowKeyValue)}
                          onClick={(e) => e.stopPropagation()}
                        />
                      </td>
                    )}
                    {columns.map((col) => (
                      <td
                        key={String(col.key)}
                        style={col.width ? { width: col.width } : undefined}
                        className={`px-6 py-4 text-body text-[color:var(--color-text-secondary)] ${col.align === "center"
                            ? "text-center"
                            : col.align === "right"
                              ? "text-right"
                              : ""
                          }`}
                      >
                        {col.render
                          ? col.render(row[col.key], row, idx)
                          : row[col.key]}
                      </td>
                    ))}
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Pagination */}
      {totalPages > 1 && (
        <div className="flex items-center justify-between">
          <p className="text-helper text-[color:var(--color-text-tertiary)]">
            共 {processedData.length} 条记录，第 {currentPage}/{totalPages} 页
          </p>
          <div className="flex gap-2">
            <button
              onClick={() => setCurrentPage(Math.max(1, currentPage - 1))}
              disabled={currentPage === 1}
              className="btn-secondary px-3 py-2 disabled:opacity-50"
            >
              上一页
            </button>
            {Array.from({ length: totalPages }).map((_, i) => {
              const pageNum = i + 1;
              const isNear =
                Math.abs(pageNum - currentPage) <= 1 ||
                pageNum === 1 ||
                pageNum === totalPages;

              if (!isNear && i > 0 && i < totalPages - 1) {
                if (i === 1) return <span key="dots">...</span>;
                return null;
              }

              return (
                <button
                  key={pageNum}
                  onClick={() => setCurrentPage(pageNum)}
                  className={`px-3 py-2 rounded-btn text-body font-semibold transition-colors ${currentPage === pageNum
                      ? "bg-[color:var(--color-primary)] text-white"
                      : "bg-white border border-[color:var(--color-border)] hover:bg-[color:var(--color-neutral)]"
                    }`}
                >
                  {pageNum}
                </button>
              );
            })}
            <button
              onClick={() =>
                setCurrentPage(Math.min(totalPages, currentPage + 1))
              }
              disabled={currentPage === totalPages}
              className="btn-secondary px-3 py-2 disabled:opacity-50"
            >
              下一页
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
