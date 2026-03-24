import { NextRequest, NextResponse } from "next/server";
import { initStore } from "../../store";

export async function POST(req: NextRequest) {
  const body = await req.json();
  initStore(body.entities ?? {}, body.start_world ?? {});
  return NextResponse.json({ ok: true });
}
