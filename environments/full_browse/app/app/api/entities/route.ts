import { NextResponse } from "next/server";
import { getEntities } from "../../store";

export const dynamic = "force-dynamic";

export async function GET() {
  return NextResponse.json(getEntities());
}
