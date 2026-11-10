import { getDocument } from "pdfjs-dist/legacy/build/pdf.mjs";

/** One positioned piece of text on a page. */
export interface TextItem {
  text: string;
  x: number;
  y: number;
}

/** A visual line: items sharing a baseline, ordered left to right. */
export interface TextLine {
  page: number;
  y: number;
  items: TextItem[];
  /** Items joined with a tab wherever there is a visible gap. */
  text: string;
}

const Y_TOLERANCE = 2.5;

/** Read every page of a PDF and return its text grouped into visual lines. */
export async function readLines(data: Uint8Array): Promise<TextLine[]> {
  const task = getDocument({
    data,
    useSystemFonts: true,
  });
  const pdf = await task.promise;
  const lines: TextLine[] = [];

  for (let pageNo = 1; pageNo <= pdf.numPages; pageNo++) {
    const page = await pdf.getPage(pageNo);
    const content = await page.getTextContent();
    const items: (TextItem & { w: number })[] = [];
    for (const it of content.items) {
      if (!("str" in it) || it.str.trim() === "") continue;
      items.push({
        text: it.str.trim(),
        x: it.transform[4],
        y: it.transform[5],
        w: it.width,
      });
    }
    // top to bottom, then left to right
    items.sort((a, b) => b.y - a.y || a.x - b.x);

    let current: (TextItem & { w: number })[] = [];
    const flush = () => {
      if (current.length === 0) return;
      current.sort((a, b) => a.x - b.x);
      let text = "";
      let prevEnd = 0;
      current.forEach((it, i) => {
        if (i > 0) text += it.x - prevEnd > 6 ? "\t" : " ";
        text += it.text;
        prevEnd = it.x + it.w;
      });
      lines.push({
        page: pageNo,
        y: current[0].y,
        items: current.map(({ text, x, y }) => ({ text, x, y })),
        text,
      });
      current = [];
    };
    for (const it of items) {
      if (current.length && Math.abs(current[0].y - it.y) > Y_TOLERANCE) flush();
      current.push(it);
    }
    flush();
  }
  await task.destroy();
  return lines;
}
