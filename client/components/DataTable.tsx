import { useState, useMemo } from "react";
import { Loader2, Inbox } from "lucide-react";
import { useTranslation } from "react-i18next";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { cn } from "@/lib/utils";

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

const alignCls = (align?: "left" | "center" | "right") =>
  align === "center" ? "text-center" : align === "right" ? "text-right" : "text-left";

export default function DataTable<T extends Record<string, any>>({
  columns,
  data,
  rowKey,
  pageSize = 10,
  selectable = false,
  onSelectionChange,
  onRowClick,
  loading = false,
  emptyMessage,
}: DataTableProps<T>) {
  const { t } = useTranslation();
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
          return sortOrder === "asc" ? aVal.localeCompare(bVal) : bVal.localeCompare(aVal);
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
      <div className="flex flex-col items-center justify-center rounded-xl border border-border bg-card py-16 text-center shadow-sm">
        <Loader2 className="mb-3 h-8 w-8 animate-spin text-primary" />
        <p className="text-sm text-muted-foreground">{t("common.loading")}</p>
      </div>
    );
  }

  if (processedData.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center rounded-xl border border-dashed border-border bg-card py-16 text-center shadow-sm">
        <Inbox className="mb-3 h-10 w-10 text-muted-foreground/50" />
        <p className="text-sm text-muted-foreground">{emptyMessage ?? t("common.none")}</p>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <div className="overflow-hidden rounded-xl border border-border bg-card shadow-sm">
        <div className="overflow-x-auto">
          <Table className="min-w-[640px]">
            <TableHeader>
              <TableRow className="hover:bg-transparent">
                {selectable && (
                  <TableHead className="w-12">
                    <Checkbox
                      checked={paginatedData.length > 0 && selectedRows.size === paginatedData.length}
                      onCheckedChange={handleSelectAll}
                    />
                  </TableHead>
                )}
                {columns.map((col) => (
                  <TableHead
                    key={String(col.key)}
                    style={col.width ? { width: col.width } : undefined}
                    className={cn("font-semibold", alignCls(col.align), col.sortable && "cursor-pointer select-none hover:bg-muted")}
                    onClick={() => col.sortable && handleSort(col.key)}
                  >
                    <span className={cn("inline-flex items-center gap-1.5", col.align === "center" ? "justify-center" : col.align === "right" ? "justify-end" : "")}>
                      {col.label}
                      {col.sortable && <span className="text-[10px] leading-none opacity-70">{sortKey === col.key ? (sortOrder === "asc" ? "▲" : "▼") : "▲▼"}</span>}
                    </span>
                  </TableHead>
                ))}
              </TableRow>
            </TableHeader>
            <TableBody>
              {paginatedData.map((row, idx) => {
                const rowKeyValue = row[rowKey];
                const isSelected = selectedRows.has(rowKeyValue);

                return (
                  <TableRow
                    key={String(rowKeyValue)}
                    className={cn("hover:bg-muted/50", isSelected && "bg-primary/5", onRowClick && "cursor-pointer")}
                    onClick={() => onRowClick && onRowClick(row)}
                  >
                    {selectable && (
                      <TableCell onClick={(e) => e.stopPropagation()}>
                        <Checkbox checked={isSelected} onCheckedChange={() => handleSelectRow(rowKeyValue)} />
                      </TableCell>
                    )}
                    {columns.map((col) => (
                      <TableCell key={String(col.key)} style={col.width ? { width: col.width } : undefined} className={alignCls(col.align)}>
                        {col.render ? col.render(row[col.key], row, idx) : row[col.key]}
                      </TableCell>
                    ))}
                  </TableRow>
                );
              })}
            </TableBody>
          </Table>
        </div>
      </div>

      {totalPages > 1 && (
        <div className="flex flex-wrap items-center justify-between gap-3">
          <p className="text-sm text-muted-foreground">
            {t("common.pageInfo", { total: processedData.length, page: currentPage, pages: totalPages })}
          </p>
          <div className="flex items-center gap-1.5">
            <Button variant="outline" size="sm" onClick={() => setCurrentPage(Math.max(1, currentPage - 1))} disabled={currentPage === 1}>
              {t("common.previousPage")}
            </Button>
            {Array.from({ length: totalPages }).map((_, i) => {
              const pageNum = i + 1;
              const isNear = Math.abs(pageNum - currentPage) <= 1 || pageNum === 1 || pageNum === totalPages;

              if (!isNear && i > 0 && i < totalPages - 1) {
                if (i === 1) return <span key="dots" className="px-1 text-muted-foreground">…</span>;
                return null;
              }

              return (
                <Button
                  key={pageNum}
                  variant={currentPage === pageNum ? "default" : "outline"}
                  size="sm"
                  className={currentPage === pageNum ? "h-9 min-w-9" : "h-9 min-w-9"}
                  onClick={() => setCurrentPage(pageNum)}
                >
                  {pageNum}
                </Button>
              );
            })}
            <Button variant="outline" size="sm" onClick={() => setCurrentPage(Math.min(totalPages, currentPage + 1))} disabled={currentPage === totalPages}>
              {t("common.nextPage")}
            </Button>
          </div>
        </div>
      )}
    </div>
  );
}
